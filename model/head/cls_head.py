import torch.nn as nn
import torch
import math


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors, cfg):
        super(ClassificationModel, self).__init__()

        self.num_classes = cfg.out_channel
        feature_size = cfg.mid_channel
        self.num_anchors = num_anchors
        self.stack_layers = cfg.get('stack_layers', 4)
        
        self.cls_convs = nn.ModuleList()
        
        for i in range(self.stack_layers):
            channel = num_features_in if i == 0 else feature_size
            self.cls_convs.append(
                nn.Conv2d(channel, feature_size, kernel_size=3, padding=1))
            self.cls_convs.append(
                nn.BatchNorm2d(feature_size))
            self.cls_convs.append(
                nn.ReLU())
            
        # self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        # self.act1 = nn.ReLU()
        self.cls_convs.append(nn.Conv2d(feature_size, num_anchors * self.num_classes, kernel_size=3, padding=1))
        self.output_act = nn.Sigmoid()

        # self.init_weights(1e-2)
    def init_weights2(self, prior_prob):
        print('init cls head')
        # for conv in self.reg_head.modules():
        b = self.output.bias.view(self.num_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.output.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        print("Finish initialize cls Head.")
    def forward(self, x):
        
        cls_feature = x
        for conv in self.cls_convs:
            cls_feature = conv(cls_feature)
        # out = self.output(cls_feature)
        out = self.output_act(cls_feature)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(x.shape[0], -1, self.num_classes)

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)