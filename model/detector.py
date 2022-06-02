import torch.nn as nn
from .backbones import build_backbone
from .head import build_head
from .necks import build_neck

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
        
    def forward(self, images, targets=None):
        """
        Args:
            images (Tensor)[B, C, H, W]: images to be processed, Shape  
        Returns:
            (list[Tensor, Tensor]): the feature extraction results for regression and classification, respectively
        """

        self.batch_size, _, self.image_h, self.image_w = images.shape
        input_size = [self.image_h, self.image_w]
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
            return [proposals_regs, proposals_clses]
    
    

            

