'目前是用于构建通用检测模型，backbone-->fpn-->head-->anchor-->label_assign-->loss'
"2021年11月24日20:31:54"
import torch
import torch.nn as nn
import math
from model.anchor.retina_anchor import Retina_Anchors
from model.anchor.build_anchor import Anchors
from model.backbones.build_backbone import build_backbone
from model.head.build_head import build_head
from model.necks.build_fpn import build_fpn
from model.post_processing.yolo_decoder import yolo_decode, clip_bboxes, nms_boxes
from model.retinanet import model


class General_detector(nn.Module):
    def __init__(self, cfg) -> None:
        super(General_detector, self).__init__()
        self.channel = 256
        # self.batch_size = cfg.TRAIN['BATCH_SIZE']
        self.num_anchors = cfg.MODEL['ANCHORS_PER_SCLAE']
        self.train_img_shape = cfg.TRAIN['TRAIN_IMG_SIZE']
        self.test_img_shape = cfg.TEST['TEST_IMG_SIZE']
        self.backbone = build_backbone(cfg)
        
        self.fpn = build_fpn(cfg.MODEL['fpn'], cfg.MODEL['out_stride'], channel_in = self.backbone.fpn_size)

        self.reg_head, self.cls_head = build_head(cfg.MODEL['head'], self.channel, self.num_anchors)

        self.anchor = Anchors()
        self.retina_anchor = Retina_Anchors()

        # for layer in self.modules():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.eval()
        self.retinanet = model.resnet18(num_classes=20, pretrained=True)
        
    def forward(self, images, type = 'train'):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                                                    [proposals_reg, proposals_cls]
        """

        self.batch_size, _, self.image_w, self.image_h = images.shape

        # large, medium, small = self.backbone(images) # {32: feature, 16: feature, 8: feature}

        # features = [large, medium, small]
        # features = self.fpn(features) # {small: feature, medium: feature, large: feature}

        # proposals_reg, proposals_cls = self.retinanet(images)
        features = self.retinanet(images)
        proposals_reg = torch.cat([self.reg_head(feature) for feature in features], dim=1)
        proposals_cls = torch.cat([self.cls_head(feature) for feature in features], dim=1)

        
        
        if type == 'test':
            # anchors = self.anchor(images)
            anchors = self.retina_anchor(images).squeeze()
            p_reg = yolo_decode(proposals_reg, anchors)
            p_cls = proposals_cls.squeeze(0)
            
            # 2. 将超出图片边界的框截掉
            p_reg = clip_bboxes(p_reg, images)
            scores, labels, boxes = nms_boxes(p_reg, p_cls)
            return [scores, labels, boxes]
        return [proposals_reg, proposals_cls]
            


    def flatten_anchors(self, anchors, feature_dim):
        """
        Args:
            anchors (list(torch.tensors)): the results of head output

        Returns: 
            anchors (list(torch.tensors)) : like [[B, N, w, h, feature_dim],...]
        """
        
        for id, item in enumerate(anchors):
            anchors[id] = item.view(self.batch_size, self.num_anchors, feature_dim, item.shape[2], item.shape[3]).permute(0, 1, 3, 4, 2)
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

