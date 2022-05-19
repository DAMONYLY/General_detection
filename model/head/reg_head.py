import torch.nn as nn
from loguru import logger
from model.utils.init_weights import *

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors, cfg):
        super(RegressionModel, self).__init__()
        
        feature_size = cfg.get('mid_channel', 256)
        self.out_channel = cfg.get('out_channel', 4)
        self.stack_layers = cfg.get('stack_layers', 4)
        self.reg_convs = nn.ModuleList()
        
        for i in range(self.stack_layers):
            channel = num_features_in if i == 0 else feature_size
            self.reg_convs.append(
                nn.Conv2d(channel, feature_size, kernel_size=3, padding=1))
            self.reg_convs.append(
                nn.BatchNorm2d(feature_size))
            self.reg_convs.append(
                nn.ReLU())

        self.reg_convs.append(nn.Conv2d(feature_size, num_anchors * self.out_channel, kernel_size=3, padding=1))
        self.num_anchors = num_anchors
    
    def init_weights(self):
        logger.info("Initialize Reg Head with config...")
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        
    def forward(self, x):

        reg_feature = x
        for conv in self.reg_convs:
            reg_feature = conv(reg_feature)
        out = reg_feature.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, self.out_channel)
