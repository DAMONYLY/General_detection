'目前是用于构建通用检测模型，backbone-->fpn-->head-->anchor-->label_assign-->loss'
"2021年11月24日20:31:54"
import torch
import torch.nn as nn
import math
from .backbones import build_backbone
from .head import build_head
from .necks import build_fpn

class General_detector(nn.Module):
    """
    General object detector for extracting image feature information only. 
    Input image, return the result after backbone, neck, head.
    """
    def __init__(self, cfg) -> None:
        super(General_detector, self).__init__()
        self.num_anchors = cfg.Model.anchors.num
        self.backbone = build_backbone(cfg)
        self.fpn = build_fpn(cfg.Model.fpn, channel_in = self.backbone.fpn_size)
        self.head = build_head(cfg.Model.head, self.fpn.channel_out, self.num_anchors)
        
    def forward(self, images):
        """
        Args:
            images (Tensor): images to be processed, Shape: [B, C, H, W] 
        Returns:
            list[Tensor, Tensor]: the feature extraction results for regression and classification, respectively
        """

        self.batch_size, _, self.image_w, self.image_h = images.shape
        features = self.backbone(images) 
        features = self.fpn(features)
        proposals_regs = []
        proposals_clses = []
        for feature in features:
            proposals_reg, proposals_cls = self.head(feature)
            
            proposals_regs.append(proposals_reg)
            proposals_clses.append(proposals_cls)
        return [proposals_regs, proposals_clses]
            

