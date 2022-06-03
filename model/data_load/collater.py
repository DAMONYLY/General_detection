import numpy as np
import torch

def simple_collater(data):
    imgs = [s['imgs'] for s in data]
    targets = [s['targets'] for s in data]
    info = [s['info'] for s in data]

    max_num_targets = max(target.shape[0] for target in targets)
    
    if max_num_targets > 0:
        tar_padded = torch.ones((len(targets), max_num_targets, 5)) * -1
        for idx, target in enumerate(targets):
            if target.shape[0] > 0:
                tar_padded[idx, :target.shape[0], :] = target
    else:
        tar_padded = torch.ones((len(target), 1, 5)) * -1
    imgs = torch.stack(imgs)
    return {'imgs': imgs, 'targets': tar_padded, 'info': info}

def multirange_collater(data):
    all_imgs = [s['imgs'] for s in data]
    # all_res_img_shape = [s['info']["res_img_shape"] for s in data]
    all_targets = [s['targets'] for s in data]
    info = [s['info'] for s in data]
    
    max_res_img_shape = [max(s) for s in zip(*[img.shape for img in all_imgs])]
    batch_res_img_shape = (len(all_imgs), *max_res_img_shape)
    batch_res_img = all_imgs[-1].new_full(batch_res_img_shape, 0)

    for i, img in enumerate(all_imgs):
        batch_res_img[i, :, :img.shape[1], :img.shape[2]] = img
        info[i]['res_img_shape'] = batch_res_img_shape
    
    max_num_targets = max(target.shape[0] for target in all_targets)
    tar_padded = all_targets[-1].new_full((len(all_targets), max_num_targets, 5), -1)
    
    if max_num_targets > 0:
        for idx, target in enumerate(all_targets):
            if target.shape[0] > 0:
                tar_padded[idx, :target.shape[0], :] = target


    return {'imgs': batch_res_img, 
            'targets': tar_padded, 
            'info': info}
