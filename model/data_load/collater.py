import numpy as np
import torch

def simple_collater(data):
    imgs = [torch.from_numpy(s['imgs']) for s in data]
    targets = [s['targets'] for s in data]
    info = [s['info'] for s in data]

    max_num_targets = max(target.shape[0] for target in targets)
    
    if max_num_targets > 0:
        tar_padded = torch.ones((len(targets), max_num_targets, 5)) * -1
        for idx, target in enumerate(targets):
            if target.shape[0] > 0:
                tar_padded[idx, :target.shape[0], :] = torch.from_numpy(target)
    else:
        tar_padded = torch.ones((len(target), 1, 5)) * -1
    imgs = torch.stack(imgs)
    return {'imgs': imgs, 'targets': tar_padded, 'info': info}
