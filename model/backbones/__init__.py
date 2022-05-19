from loguru import logger
from .darknet53 import Darknet53
from .resnet import *
from .shufflenetv2 import ShuffleNetV2

def build_backbone(cfg):
    name = cfg.Model.backbone.name
    pre_train = getattr(cfg.Model.backbone, 'pretrain', False)
    if name == 'Darknet53':
        model = Darknet53()
        if cfg.pre_train:
            model.load_darknet_weights(cfg.weight_path)
        return Darknet53()
    elif name.lower() == 'resnet':
        assert 'depth' in cfg.Model.backbone, 'must give the depth of resnet model in cfg file.'
        depth = cfg.Model.backbone.depth
        if depth == 18:
            model = resnet18()
        elif depth == 34:
            model = resnet34()
        elif depth == 50:
            model = resnet50()
        elif depth == 101:
            model = resnet101()
        elif depth == 152:
            model = resnet152()
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
        if pre_train == True:
            name = name.lower() + str(depth)
            logger.info(f'=> loading {name} backbone from {model_urls[name]}')
            model.load_state_dict(model_zoo.load_url(model_urls[name], model_dir='.'), strict=False)
        else:
            logger.warning('Not loading the pretained model...')
        return model
    elif name.lower() == 'shufflenetv2':
        size = getattr(cfg.Model.backbone, 'model_size', '1.0x')
        model = ShuffleNetV2(model_size=size)
        if pre_train == True:
            model._initialize_weights(pretrain=pre_train)
        else:
            logger.warning('Not loading the pretained model...')
        return model
    else:
        raise NotImplementedError(f'{name} model not support yet...')

