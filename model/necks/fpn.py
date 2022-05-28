import torch.nn as nn
from loguru import logger
from model.utils.init_weights import * 


class FPN(nn.Module):
    """
    the FPN of retinanet.
    """
    def __init__(self, channel_in = 256, channel_out = 256):
        super(FPN, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.channel_out = channel_out
        self.P5_1 = nn.Conv2d(channel_in[2], channel_out, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(channel_in[1], channel_out, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(channel_in[0], channel_out, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(channel_in[2], channel_out, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=2, padding=1)

    def init_weights(self):
        logger.info('initialize Fpn with config...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")
                
    def forward(self, inputs):
        C3, C4, C5 = inputs # large, medium, small

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
