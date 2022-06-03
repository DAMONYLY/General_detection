import torch.nn as nn
import torch
import torchvision
from .backbones import build_backbone
from .head import build_head
from .necks import build_neck
from .utils import clip_bboxes

class General_detector(nn.Module):
    """
    General object detector for extracting image feature information only. 
    Input image, return the result after backbone, neck, head.
    """
    def __init__(self, cfg) -> None:
        super(General_detector, self).__init__()
        self.num_anchors = cfg.Model.anchors.num
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg.Model.neck, channel_in = self.backbone.neck_size)
        self.head = build_head(cfg, self.neck.channel_out, cfg.Model.anchors.num)
        self.conf_thre = 0.05
        self.nms_thre = cfg.Data.test.nms_thre

    def forward(self, images, targets=None):
        """
        Args:
            images (Tensor)[B, C, H, W]: images to be processed, Shape  
        Returns:
            (list[Tensor, Tensor]): the feature extraction results for regression and classification, respectively
        """

        batch_size, _, image_h, image_w = images.shape
        input_size = [image_h, image_w]
        features = self.backbone(images) 
        features = self.neck(features)
        proposals_regs = []
        proposals_clses = []
        
        for feature in features:
            proposals_reg, proposals_cls = self.head(feature)
            proposals_regs.append(proposals_reg)
            proposals_clses.append(proposals_cls)
        if self.training:
            loss = self.head.loss_calculater([proposals_regs, proposals_clses], targets, input_size)
            return loss
        else:
            # post_process
            proposals_regs = torch.cat(proposals_regs, dim=1)
            proposals_clses = torch.cat(proposals_clses, dim=1)
            anchors = self.head.anchors(input_size, proposals_regs.device, proposals_regs.dtype)

            assert len(proposals_regs) == len(proposals_clses)
            output = [None for _ in range(batch_size)]
            for id in range(batch_size):
                proposals_reg = proposals_regs[id]
                proposals_cls = proposals_clses[id]

                if proposals_reg.size(1) > 4:
                    p_obj = proposals_reg[:, 4]
                else:
                    p_obj = proposals_reg.new_full((proposals_reg.size(0), ), 1)
                
                p_reg = proposals_reg[:, :4]
                # change pred_reg to xyxy form
                p_reg = self.head.decode(p_reg, anchors)
                p_reg = clip_bboxes(p_reg, input_size)

                class_conf, class_pred = torch.max(proposals_cls, 1, keepdim=True)
                conf_mask = (p_obj * class_conf.squeeze() >= self.conf_thre).squeeze()
                # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                detections = torch.cat((p_reg, p_obj.unsqueeze(1), class_conf, class_pred.float()), 1)
                detections = detections[conf_mask]

                if not detections.size(0):
                    continue

                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    self.nms_thre,
                )

                detections = detections[nms_out_index]
                if output[id] is None:
                    output[id] = detections
                else:
                    output[id] = torch.cat((output[id], detections))

            return output

    
    

            

