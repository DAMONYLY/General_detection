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
        self.reg_head, self.cls_head = build_head(cfg.Model.head, self.fpn.channel_out, self.num_anchors)
        # self.init_head(1e-2)

    def init_head(self, prior_prob):
        print('init reg and cls head')
        for conv in self.reg_head.modules():
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.cls_head:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        
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
        proposals_regs = torch.cat([self.reg_head(feature) for feature in features], dim=1)
        proposals_clses = torch.cat([self.cls_head(feature) for feature in features], dim=1)

        return [proposals_regs, proposals_clses]
            

