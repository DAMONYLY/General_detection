import numpy as np
import math
import random
import cv2
import torch

class Pipeline:
    def __init__(self, args):
        self.augment = []
        for i, augs in enumerate(args):
            self.augment.append(eval(augs)(args[augs]))

    def __call__(self, sample):
        for augs in self.augment:
            sample = augs(sample)
        return sample

class resize_aug:
    def __init__(self, args) -> None:
        self.input_size = args.input_size
        self.keep_ratio = args.keep_ratio
        
    def __call__(self, sample):

        img = sample['imgs'].copy()
        targets = sample['targets'].copy()

        resize_size = random.randint(*self.input_size)
        resize_size = math.ceil(resize_size / 32) * 32

        padded_img = np.ones((resize_size, resize_size, 3), dtype=np.uint8) * 114

        # 将原图片尺寸转成预设的输入图片尺寸
        r = min(resize_size / img.shape[0], resize_size / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        if self.keep_ratio:
            padded_img = resized_img.transpose(2, 0, 1)
        else:
            padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
            padded_img = padded_img.transpose(2, 0, 1)
            padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) # 返回连续的数组在内存中，same as contiguous()
        
        targets[:, :4] = targets[:, :4] * r

        sample['imgs'] = padded_img
        sample['targets'] = targets
        sample['info']['res_img_shape'] = padded_img.shape
        sample['info']['scale'] = r
        return sample

class hsv_aug:
    def __init__(self, args):
        self.hsv_prob = args.prob
        self.hgain = 5 if not hasattr(args, 'hgain') else args.hgain
        self.sgain = 30 if not hasattr(args, 'sgain') else args.sgain
        self.vgain = 30 if not hasattr(args, 'vgain') else args.vgain

    def __call__(self, sample):
        if random.random() < self.hsv_prob:
            img = sample['imgs'].copy()
            hsv_augs = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]  # random gains
            hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
            hsv_augs = hsv_augs.astype(np.int16)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

            img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
            img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
            img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

            hsv_img = cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed
            sample['imgs'] = hsv_img
        return sample


class flip_aug:
    def __init__(self, args):
        self.flip_prob = args.prob
        
    def __call__(self, sample):
        if random.random() < self.flip_prob:
            image = sample['imgs'].copy()
            boxes = sample['targets'].copy()
            _, width, _ = image.shape
            image = image[:, ::-1]
            boxes[:, [0,2]] = width - boxes[:, [2,0]]
            sample['imgs'] = image
            sample['targets'] = boxes
        return sample


class norm_aug:
    def __init__(self, args):
        self.mean = np.array(args.mean).reshape(1,-1)
        self.std = np.array(args.std).reshape(1, -1)
        
    def __call__(self, sample):
        image = sample['imgs'].copy()
        if image.shape[2] != 3:
            image = image.transpose(1,2,0)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.subtract(image, self.mean)
        stdinv = 1 / self.std
        image=cv2.multiply(image, stdinv).transpose(2,0,1)
        sample['imgs'] = image
        return sample

class to_tensor_aug:
    def __init__(self, args):
        pass
    def __call__(self, sample):
        image = sample['imgs'].copy()
        targets = sample['targets'].copy()
        
        image = torch.from_numpy(image).contiguous()
        targets = torch.from_numpy(targets)

        sample['imgs'] = image.float()
        sample['targets'] = targets.float()
        return sample