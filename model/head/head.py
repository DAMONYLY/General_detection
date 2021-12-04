import torch.nn as nn
from ..layers.conv_module import Convolutional

class Head(nn.Module):
    def __init__(self, channel_in, channel_mid, channel_out, num_anchor):
        """
        
        build head include reg, cls, obj

        Arguments:
            channel_in (int): the feature channel shape of the out of fpn
            channel_mid (int): the channel shape in head Conv, usually same with channel_in
            channel_out (list[] or int): the out channel of head, eg. [reg, cls] or [reg, cls, obj]

        """
        super(Head, self).__init__()
        reg_channel, cls_channel = channel_out
        self.reg_s = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            # Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            # kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=reg_channel * num_anchor, 
                            kernel_size=1, stride=1, pad=0),
        )

        self.cls_s = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            # Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            # kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=cls_channel * num_anchor, 
                            kernel_size=1, stride=1, pad=0),
        )

        self.reg_m = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            # Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            # kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=reg_channel * num_anchor, 
                            kernel_size=1, stride=1, pad=0),
        )

        self.cls_m = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            # Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            # kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=cls_channel * num_anchor, 
                            kernel_size=1, stride=1, pad=0),
        )

        self.reg_l = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            # Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            # kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=reg_channel * num_anchor, 
                            kernel_size=1, stride=1, pad=0),
        )

        self.cls_l = nn.Sequential(
            Convolutional(filters_in=channel_in, filters_out=channel_mid, 
                            kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            # Convolutional(filters_in=channel_mid, filters_out=channel_mid, 
                            # kernel_size=3, stride=1, pad=1, norm='bn', activate='relu'),
            Convolutional(filters_in=channel_mid, filters_out=cls_channel * num_anchor, 
                            kernel_size=1, stride=1, pad=0),
        )
    def forward(self, inputs): # [small, medium, large]
        """Return feature maps.

        Arguments:
            inputs (list[torch.Tensor]): the out feature of fpn. [small, medium, large]

        Returns:
            list(Tensor): list of head feature maps, eg. [reg_s, reg_m, reg_l], [cls_s, cls_m, cls_l]
        """
        s, m, l = inputs

        reg_s = self.reg_s(s)
        cls_s = self.cls_s(s)

        reg_m = self.reg_m(m)
        cls_m = self.cls_m(m)

        reg_l = self.reg_l(l)
        cls_l = self.cls_l(l)
        return [reg_s, reg_m, reg_l], [cls_s, cls_m, cls_l]