
from .retinanet_sample import Retinanet_sampler

def build_sampler(cfg):
    if cfg.Model.sample.name == 'Retinanet':
        return Retinanet_sampler(cfg.Model.sample.bbox_coder, cfg.Classes.num)
    else:
        raise NotImplementedError