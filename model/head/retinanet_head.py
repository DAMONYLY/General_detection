import torch.nn as nn
from loguru import logger
from model.utils.init_weights import *
from model.layers import ConvModule

class Retinanethead(nn.Module):
    def __init__(self, num_features_in, num_anchors, cfg):
        super(Retinanethead, self).__init__()
        
        reg_feature_size = cfg.reg_head.get('mid_channel', 256)
        self.reg_out_channel = cfg.reg_head.get('out_channel', 4)
        self.reg_stack_layers = cfg.reg_head.get('stack_layers', 4)
        
        cls_feature_size = cfg.cls_head.get('mid_channel', 256)
        self.cls_out_channel = cfg.cls_head.out_channel
        self.cls_stack_layers = cfg.cls_head.get('stack_layers', 4)   
        
        self.reg_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.reg_stack_layers):
            reg_channel = num_features_in if i == 0 else reg_feature_size
            self.reg_convs.append(ConvModule(filters_in=reg_channel,
                                             filters_out=reg_feature_size,
                                             kernel_size=3,
                                             stride=1,
                                             pad=1,
                                             norm='bn',
                                             activate='relu'
                                             )
                                  )
        for i in range(self.cls_stack_layers):
            cls_channel = num_features_in if i == 0 else cls_feature_size
            self.cls_convs.append(ConvModule(filters_in=cls_channel,
                                             filters_out=cls_feature_size,
                                             kernel_size=3,
                                             stride=1,
                                             pad=1,
                                             norm='bn',
                                             activate='relu'
                                             )
                                  )
            
        self.reg_head = ConvModule(reg_feature_size, num_anchors * self.reg_out_channel, kernel_size=3, pad=1)
        
        self.cls_head = ConvModule(cls_feature_size, num_anchors * self.cls_out_channel, kernel_size=3, pad=1, activate='sigmoid')
        self.num_anchors = num_anchors
    
    def init_weights(self):
        logger.info("=> Initialize Head ...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
    def forward(self, x):
        b, _, _, _ = x.shape
        reg_feature = x
        cls_feature = x
        for reg_conv, cls_conv in zip(self.reg_convs, self.cls_convs):
            reg_feature = reg_conv(reg_feature)
            cls_feature = cls_conv(cls_feature)
        reg_out = self.reg_head(reg_feature)
        cls_out = self.cls_head(cls_feature)
        reg_out = reg_out.permute(0, 2, 3, 1).contiguous().view(b, -1, self.reg_out_channel)
        cls_out = cls_out.permute(0, 2, 3, 1).contiguous().view(b, -1, self.cls_out_channel)
        
        return reg_out, cls_out
