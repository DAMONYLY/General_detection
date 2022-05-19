from .datasets import CocoDataset
from .sampler import InfiniteSampler
from .collater import simple_collater

from torch.utils.data import BatchSampler, SequentialSampler
from torch.utils.data import DataLoader

def build_train_dataloader(cfg, batch_size, num_workers, seed=0):
    dataset = CocoDataset(cfg.dataset_path, set_name=cfg.set_name, pipeline=cfg.pipeline)

    sampler = InfiniteSampler(len(dataset), shuffle=True, seed=seed)

    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)

    dataloader_kwargs = {"num_workers": num_workers, 
                         "pin_memory": True,
                         "batch_sampler": batch_sampler,
                         "collate_fn": simple_collater
                         }

    train_loader = DataLoader(dataset, **dataloader_kwargs)

    return train_loader

def build_val_dataloader(cfg, batch_size, num_workers):
    valdataset = CocoDataset(cfg.dataset_path, set_name=cfg.set_name, pipeline=cfg.pipeline)
    sampler = SequentialSampler(valdataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    dataloader_kwargs = {"num_workers": num_workers,
                         "pin_memory": True,
                         "batch_sampler": batch_sampler,
                         "collate_fn": simple_collater,
                         }

    val_loader = DataLoader(valdataset, **dataloader_kwargs)

    return val_loader
