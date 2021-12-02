'目前是用于构建通用检测模型，backbone-->fpn-->head-->anchor-->label_assign-->loss'
"2021年11月24日20:31:54"
import torch
import torch.nn as nn
from model.anchor.build_anchor import Anchors
from model.backbones.build_backbone import build_backbone
from model.head.build_head import build_head
from model.loss.build_loss import build_loss
from model.metrics.build_metrics import build_metrics
from model.necks.build_fpn import build_fpn

class General_detector(nn.Module):
    def __init__(self, cfg) -> None:
        super(General_detector, self).__init__()
        self.channel = 256
        self.batch_size = cfg.TRAIN['BATCH_SIZE']
        self.backbone = build_backbone(cfg.MODEL['backbone'])
        
        self.fpn = build_fpn(cfg.MODEL['fpn'], cfg.MODEL['out_stride'])

        self.head = build_head(cfg.MODEL['head'], self.channel, cfg.MODEL['ANCHORS_PER_SCLAE'])

        self.anchors = Anchors()

        self.label_assign = build_metrics(cfg.MODEL['metrics'])
        
        self.loss = build_loss(cfg.MODEL['loss'], cfg)
        
    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if targets is None:
            return 0
        features8, features16, features32 = self.backbone(images) # {32: feature, 16: feature, 8: feature}
        features = [features32, features16, features8]
        features = self.fpn([features32, features16, features8]) # {32: feature, 16: feature, 8: feature}

        proposals_reg, proposals_cls = self.head(features) # {32: proposal, 16: proposal, 8: proposal}

        proposals_reg = self.flatten_anchors(proposals_reg, 5)
        proposals_cls = self.flatten_anchors(proposals_cls, 20)
        anchors = self.anchors(images)

        label_assign, cls_label, reg_label = self.label_assign(anchors, targets, proposals_reg, proposals_cls)

        losses, losses_xy, losses_wh, losses_cls = self.loss(label_assign, proposals_cls, cls_label, proposals_reg, reg_label) # reg_loss, cls_loss, conf_loss

        return losses, losses_xy, losses_wh, losses_cls

    def flatten_anchors(self, anchors, feature_dim):
        """
        Args:
            anchors (list(torch.tensors)): the results of head output

        Returns: 
            anchors (torch.tensors) : like [B, N, feature_dim]
        """

        for id, item in enumerate(anchors):
            anchors[id] = item.view(self.batch_size, feature_dim, -1)
        anchors = torch.cat(anchors, dim=2).transpose(1, 2)

        return anchors

    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"
        import torch
        import numpy as np
        from model.layers.conv_module import Convolutional
        print("load darknet weights : ", weight_file)

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                print("loading weight {}".format(conv_layer))

