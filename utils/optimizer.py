import copy
import torch
from .lr_scheduler import *
def build_optimizer(cfg, dataset_length, model):
    optimizer_cfg = cfg.Schedule.optimizer
    name = optimizer_cfg.pop("name")
    build_optimizer = getattr(torch.optim, name)
    optimizer = build_optimizer(params=model.parameters(), **optimizer_cfg)

    schedule_cfg = cfg.Schedule.lr_schedule
    schedule_cfg['warmup'] *= dataset_length
    schedule_cfg['milestones'] = (np.array(schedule_cfg['milestones']) + 1) * dataset_length
    name = schedule_cfg.pop("name")
    if name == 'MultiStep_LR':
        lr_scheduler = MultiStep_LR(optimizer=optimizer, **schedule_cfg)
    elif name == 'CosineDecay_LR':
        lr_scheduler = CosineDecay_LR(optimizer=optimizer, **schedule_cfg)
    
    return optimizer, lr_scheduler