
from .darknet53 import Darknet53
from .resnet import *
from .shufflenetv2 import ShuffleNetV2

def build_backbone(cfg):
    name = cfg.Model.backbone.name
    pre_train = getattr(cfg, 'pre_train', False)
    if name == 'Darknet53':
        model = Darknet53()
        if cfg.pre_train:
            model.load_darknet_weights(cfg.weight_path)
        return Darknet53()
    elif name.lower() == 'resnet':
        assert 'depth' in cfg.Model.backbone, 'must give the depth of resnet model in cfg file.'
        depth = cfg.Model.backbone.depth
    
        if depth == 18:
            return resnet18(pretrained=pre_train)
        elif depth == 34:
            return resnet34(pretrained=pre_train)
        elif depth == 50:
            return resnet50(pretrained=pre_train)
        elif depth == 101:
            return resnet101(pretrained=pre_train)
        elif depth == 152:
            return resnet152(pretrained=pre_train)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    elif name.lower() == 'shufflenetv2':
        return ShuffleNetV2(pretrained=pre_train)
    else:
        raise NotImplementedError