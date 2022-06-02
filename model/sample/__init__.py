from .retinanet_sample import Retinanet_sampler
from .free_sample import Free_sampler

def build_sampler(cfg):
    if cfg.Model.sample.name == 'Retinanet':
        return Retinanet_sampler(cfg.Model.sample.bbox_coder, cfg.Classes.num)
    elif cfg.Model.sample.name == 'free':
        return Free_sampler(cfg.Model.sample.bbox_coder, cfg.Classes.num)
    else:
        raise NotImplementedError