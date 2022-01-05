
from .darknet53 import Darknet53
from .resnet import *

def build_backbone(cfg):
    name = cfg.MODEL['backbone']
    if name == 'Darknet53':
        model = Darknet53()
        if cfg.pre_train:
            model.load_darknet_weights(cfg.weight_path)
        return Darknet53()
    elif name.lower() == 'resnet':
        assert 'depth' in cfg.MODEL, 'must give the depth of resnet model in cfg file.'
        depth = cfg.MODEL['depth']
        num_classes = 20
        if depth == 18:
            return resnet18(num_classes, pretrained=cfg.pre_train)
        elif depth == 34:
            return resnet34(num_classes, pretrained=cfg.pre_train)
        elif depth == 50:
            return resnet50(num_classes, pretrained=cfg.pre_train)
        elif depth == 101:
            return resnet101(num_classes, pretrained=cfg.pre_train)
        elif depth == 152:
            return resnet152(num_classes, pretrained=cfg.pre_train)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    else:
        raise NotImplementedError