import torch
import torch.nn as nn
import torch.nn.functional as F
from .activate import *


norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    'sigmoid': nn.Sigmoid,
    "HardMish": HardMish}


class ConvModule(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, 
                 stride=1, pad=0, norm=None, activate=None,
                 ):
        super(ConvModule, self).__init__()

        self.norm = norm
        self.activate = activate
        self.out_channels = filters_out
        self.conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                stride=stride, padding=pad, bias=not norm)
        if norm:
            assert norm in norm_name.keys(), f'Unsupport norm type, now only support {norm_name.keys()}'
            self.norm = norm_name[norm](filters_out)

        if activate:
            assert activate in activate_name.keys(), f'Unsupport activate type, now only support {activate_name.keys()}'
            self.activate = activate_name[activate]()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activate:
            x = self.activate(x)
        return x
