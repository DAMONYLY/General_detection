'''
2021年11月29日19:59:10
通用的FPN
'''
import torch.nn as nn
import torch.nn.functional as F
from ..layers.conv_module import Convolutional
import math


def make_layer(stride, channel_in, channel_out):
    m = nn.Sequential()
    m.add_module(str(stride) + '_1',
            Convolutional(filters_in=channel_in, filters_out=channel_out, kernel_size=1, stride=1, pad=0, norm="bn",activate="relu"))
    m.add_module(str(stride) + '_upsample', nn.Upsample(scale_factor=2, mode='nearest'))
    m.add_module(str(stride) + '_2',
            Convolutional(filters_in=channel_in, filters_out=channel_out, kernel_size=3, stride=1, pad=1, norm="bn",activate="relu"))
            
    return m

class FPN(nn.Module):
    """
    the FPN of retinanet.
    TODO: 实现根据strides 动态构建FPN
    """
    def __init__(self, strides, channel_in = 256, channel_out = 256):
        super(FPN, self).__init__()


        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(channel_in[0], channel_out, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(channel_in[1], channel_out, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(channel_in[2], channel_out, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(channel_in[0], channel_out, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=2, padding=1)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")
    def forward(self, inputs):
        C5, C4, C3 = inputs # large, medium, small

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)