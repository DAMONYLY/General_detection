from .datasets import CocoDataset
from .sampler import InfiniteSampler
from .collater import simple_collater

from torch.utils.data import BatchSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist

def build_train_dataloader(cfg, batch_size, num_workers, is_distributed, seed=0):
    dataset = CocoDataset(cfg.dataset_path, set_name=cfg.set_name, pipeline=cfg.pipeline)

    sampler = InfiniteSampler(len(dataset), shuffle=True, seed=seed if seed is not None else 0)
    if is_distributed:
        batch_size = batch_size // dist.get_world_size()

    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)

    dataloader_kwargs = {"num_workers": num_workers, 
                         "pin_memory": True,
                         "batch_sampler": batch_sampler,
                         "collate_fn": simple_collater
                         }

    train_loader = DataLoader(dataset, **dataloader_kwargs)

    return train_loader

def build_val_dataloader(cfg, batch_size, num_workers, is_distributed):
    valdataset = CocoDataset(cfg.dataset_path, set_name=cfg.set_name, pipeline=cfg.pipeline)

    if is_distributed:
        batch_size = batch_size // dist.get_world_size()
        sampler = DistributedSampler(
            valdataset, shuffle=False
        )
    else:
        sampler = SequentialSampler(valdataset)

    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    dataloader_kwargs = {"num_workers": num_workers,
                         "pin_memory": True,
                         "batch_sampler": batch_sampler,
                         "collate_fn": simple_collater,
                         }

    val_loader = DataLoader(valdataset, **dataloader_kwargs)

    return val_loader
