import torch.nn as nn
import torch
from ..layers.conv_module import Convolutional

class Head(nn.Module):
    def __init__(self, channel_in, channel_mid, channel_out, num_anchor):
        """
        
        build head include reg, cls, obj

        Arguments:
            channel_in (list(int)): different number of each level fpn out feature channel. 
                            [level3_channel, level4_channel, level5_channel]
            channel_mid (int): the channel number in head Conv, usually same.
                            256
            channel_out (list[] or int): the out channel of head, eg. [reg, cls] or [reg, cls, obj]

        """
        super(Head, self).__init__()
        self.reg_channel, self.cls_channel = channel_out
        self.num_anchors = num_anchor
        self.reg_s = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=self.reg_channel * num_anchor, 
                            kernel_size=3, stride=1, pad=1),
        )

        self.cls_s = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=self.cls_channel * num_anchor, 
                            kernel_size=3, stride=1, pad=1),
        )

        self.reg_m = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=self.reg_channel * num_anchor, 
                            kernel_size=3, stride=1, pad=1),
        )

        self.cls_m = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=self.cls_channel * num_anchor, 
                            kernel_size=3, stride=1, pad=1),
        )

        self.reg_l = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),              
            Convolutional(filters_in=channel_mid, filters_out=self.reg_channel * num_anchor, 
                            kernel_size=3, stride=1, pad=1),
        )

        self.cls_l = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=self.cls_channel * num_anchor, 
                            kernel_size=3, stride=1, pad=1),
        )
    def forward(self, inputs): # [large, medium, small]
        """Return feature maps.

        Arguments:
            inputs (list[torch.Tensor]): the out feature of fpn. [large, medium, small]

        Returns:
            list(Tensor): list of head feature maps, eg. [reg_l, reg_m, reg_s], [cls_l, cls_m, cls_s]
                        shape: [B, num_anchor * outdim, w, h]
        """
        l, m, s = inputs
        reg_s = self.reg_s(s)
        cls_s = self.cls_s(s)

        reg_m = self.reg_m(m)
        cls_m = self.cls_m(m)

        reg_l = self.reg_l(l)
        cls_l = self.cls_l(l)

        # change [B, num_anchor * outdim, w, h] to [B, num_anchor, w, h, out_dim]
        # reg_s = reg_s.view(batch_size, self.num_anchors, self.reg_channel, reg_s.shape[2], reg_s.shape[3])

        regression = torch.cat([self.reshape_feature(feature) for feature in [reg_s, reg_m, reg_l]], dim=1)
        classification = torch.cat([self.reshape_feature(feature) for feature in [cls_s, cls_m, cls_l]], dim=1)
        # return torch.cat([reg_s, reg_m, reg_l], [cls_l, cls_m, cls_s]
        return regression, classification

    def reshape_feature(self, feature):
        B, C, W, H = feature.shape
        feature = feature.permute(0, 2, 3, 1)
        feature = feature.contiguous().view(B, -1, C//self.num_anchors)
        return feature