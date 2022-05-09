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
            #print(annot.shape)
            if target.shape[0] > 0:
                tar_padded[idx, :target.shape[0], :] = torch.from_numpy(target)
    else:
        tar_padded = torch.ones((len(target), 1, 5)) * -1
    imgs = torch.stack(imgs)
    return {'imgs': imgs, 'targets': tar_padded, 'info': info}


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
