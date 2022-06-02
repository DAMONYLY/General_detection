from typing import Dict
import numpy as np
import math
import random
import cv2

class Pipeline:
    def __init__(self, args: Dict) -> None:
        self.shape_transform = Shape_transform(**args)
        self.augment = Augment(**args)
        

    def __call__(self, sample: Dict):
        sample = self.shape_transform(sample)
        # sample = self.augment(sample)
        return sample


class Shape_transform:
    def __init__(self, input_size, keep_ratio = False, flip_prob = 0.0, hsv_prob = 0.0, **kwargs):
        # 1. flip
        # 2. hsv_trans
        # 3. resize
        self.input_size = input_size
        self.keep_ratio = keep_ratio
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        
    def __call__(self, sample):
        ori_img = sample['imgs']
        targets = sample['targets']
        ori_bbox = targets[:, :4]
        ori_h, ori_w, _ = ori_img.shape

        # hsv
        if random.random() < self.hsv_prob:
            img = hsv_augment(ori_img)
        else:
            img = ori_img
        # flip
        if random.random() < self.flip_prob:
            img, box = flip_augment(img, ori_bbox)
        else:
            box = ori_bbox
        # resize
        resized_img, r = resize_augment(img, self.input_size, self.keep_ratio)
        box *= r

        image = resized_img.astype(np.float32).transpose(1,2,0)
        # if(self.to_rgb): #如果是采用caffe风格的预训练，则无需转化为RGB格式，如果是pytorch风格的预训练，需要转化为RGB
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.subtract(image, np.array([123.675, 116.28, 103.53]).reshape(1,-1))
        stdinv = 1/np.array([58.395, 57.12, 57.375]).reshape(1,-1)
        resized_img=cv2.multiply(image, stdinv).transpose(2,0,1)

        targets[:, :4] = box
        _, res_h, res_w = resized_img.shape
        sample['imgs'] = resized_img
        sample['targets'] = targets
        sample['info']['res_img_info'] = [res_h, res_w]
        sample['info']['scale'] = r

        return sample
        

class Augment:
    def __init__(self, **kwargs) -> None:
        pass
    def __call__(self, *args):
        pass


def hsv_augment(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    hsv_img = cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    return hsv_img

def flip_augment(image, boxes):
    _, width, _ = image.shape
    image = image[:, ::-1]
    boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes

def resize_augment(img, input_size, keep_ratio=False, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    # 将原图片尺寸转成预设的输入图片尺寸
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    if keep_ratio:
        resized_img = resized_img.transpose(swap)
        return resized_img, r
    
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) # 返回连续的数组在内存中，same as contiguous()
    return padded_img, r