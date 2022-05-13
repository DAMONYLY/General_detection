import torch.nn as nn
import torch
import math

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
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # self.init_weights(1e-2)

    def init_weights2(self, prior_prob):
        print('init reg head')
        # for conv in self.reg_head.modules():
        b = self.output.bias.view(self.num_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.output.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
    def init_weights(self):
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        print("Finish initialize reg Head.")



    def forward(self, x):

        reg_feature = x
        for conv in self.reg_convs:
            reg_feature = conv(reg_feature)
        # out = self.output(reg_feature)
        out = reg_feature.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, self.out_channel)

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)