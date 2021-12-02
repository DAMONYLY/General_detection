import torch.nn as nn

from model.backbones.darknet53 import Darknet53
def build_backbone(name):
    if name == 'Darknet53':
        return Darknet53()
    elif name == 'Resnet':
        raise NotImplementedError