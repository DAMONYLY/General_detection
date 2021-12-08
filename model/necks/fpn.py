'''
2021年11月29日19:59:10
通用的FPN
'''
import torch.nn as nn
import torch.nn.functional as F
from ..layers.conv_module import Convolutional



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

        # layers = []
        # for stride in strides:
            # layers.append(make_layer(stride))

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = Convolutional(filters_in=channel_in[0], filters_out=channel_out, kernel_size=1, stride=1, pad=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = Convolutional(filters_in=channel_out, filters_out=channel_out, kernel_size=3, stride=1, pad=1, norm="bn",activate="relu")

        # add P5 elementwise to C4
        self.P4_1 = Convolutional(filters_in=channel_in[1], filters_out=channel_out, kernel_size=1, stride=1, pad=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = Convolutional(filters_in=channel_out, filters_out=channel_out, kernel_size=3, stride=1, pad=1, norm="bn",activate="relu")

        # add P4 elementwise to C3
        self.P3_1 = Convolutional(filters_in=channel_in[2], filters_out=channel_out, kernel_size=1, stride=1, pad=0)
        self.P3_2 = Convolutional(filters_in=channel_out, filters_out=channel_out, kernel_size=3, stride=1, pad=1, norm="bn",activate="relu")

    # def forward(self, x0, x1, x2):  # large, medium, small
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


        return [P5_x, P4_x, P3_x] # large, medium, small
