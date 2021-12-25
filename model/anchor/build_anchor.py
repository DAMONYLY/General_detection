'''
2021年11月29日16:25:10

用于产生anchor
'''


import numpy as np
from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import math

class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [5, 4, 3]
        if strides is None:
            # self.strides = [2 ** x for x in self.pyramid_levels]
            self.strides = [32, 16, 8]
        if sizes is None:
            self.sizes = [2 ** (x + 3) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            # self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
            self.scales = np.array([2 ** 0])
    def make_base_anchor(self, base_size=16, ratios=None, scales=None):
        base_anchors = []
        for ratio in ratios:
            for scale in scales:
                base_anchors.append([-scale * math.sqrt(ratio), -scale / math.sqrt(ratio), scale * math.sqrt(ratio), scale / math.sqrt(ratio)])
        base_anchors = (torch.tensor(base_anchors) * (base_size / 2))
        return base_anchors
    
    def generate_anchor_single_level(self, shape, stride, anchors):
        shift_x = (torch.arange(0, shape[1]) + 0.5) * stride
        shift_y = (torch.arange(0, shape[0]) + 0.5) * stride

        shift_x, shift_y = torch.meshgrid([shift_x, shift_y])
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y), dim=1)
        center_x = shifts[:, 0].unsqueeze(1)
        center_y = shifts[:, 1].unsqueeze(1)
        grids = torch.cat([center_x, center_y, center_x, center_y], dim=1)
        all_anchors = grids[None, :, :] + anchors[:, None, :]
        # all_anchors = all_anchors.reshape(-1, 4)
        return all_anchors

    def forward(self, image, only_anchors = False):
        """
        Args:
            image (torch.Tensor(B, C, W, H)): the image need to process


        Returns:
            anchors (torch.Tensor(levels, num_anchors, 4)): anchor based on input image size with xyxy (bottom left, top right)
            X = (size/stride1)**2 *num_anchor + (size/stride2)**2 *num_anchor + (size/stride3)**2 *num_anchor
            large + medium + small
        """
        if isinstance(image, int):
            image_shape = torch.tensor([image, image])
        elif isinstance(image, torch.Tensor):
            image_shape = image.shape[2:4]
        image_shape = np.array(image_shape)
        image_shapes = [image_shape // s for s in self.strides]

        
        all_anchors = []

        for idx, level in enumerate(self.pyramid_levels):

            base_anchors = self.make_base_anchor(self.sizes[idx], self.ratios, self.scales)
            if only_anchors:
                one_anchors = base_anchors
            else:
                one_anchors = self.generate_anchor_single_level(image_shapes[idx], self.strides[idx], base_anchors)
            all_anchors.append(one_anchors)

        return all_anchors

if __name__ == "__main__":
    gen = Anchors()
    in_img = torch.randn(12, 3, 416, 416)
    a = gen(in_img)
    import sys
    sys.path.append('/raid/yaoliangyong/General_detection/')
    # import matplotlib.pyplot as plt
    import cv2
    from utils.visualize import visualize_boxes
    # imshow = cv2.imread('/raid/yaoliangyong/General_detection/000001.jpg')
    # imshow = cv2.resize(imshow, (320, 416))

    for lever in range(len(a)):
        for anchor in range(a[lever].shape[0]):
            imshow = cv2.imread('/raid/yaoliangyong/General_detection/000001.jpg')
            test = cv2.imread('/raid/yaoliangyong/General_detection/lever_0_shape_0_.jpg')
            imshow = cv2.resize(imshow, (416, 416))
            _boxes = a[lever][anchor]
            if len(_boxes.shape) == 1:
                _boxes = _boxes.unsqueeze(0)
            _boxes = _boxes.numpy()
            sum_b = len(_boxes)
            _labels = np.array([1] *sum_b)
            _probs = (np.arange(0, sum_b) + 0.1)/10
            # _probs = np.ones((sum_b, 1))
            cateNames = [
                "person",
                "bicycle",
                "car",
                "motorbike",
                "aeroplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "sofa",
                "pottedplant",
                "bed",
                "diningtable",
                "toilet",
                "tvmonitor",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush"
            ]
            visualize_boxes(image=imshow, boxes=_boxes, labels=_labels, probs=_probs, class_labels=cateNames)
            name = 'lever_' + str(lever) + '_shape_' + str(anchor) + '_.jpg'
            cv2.imwrite(name, imshow)
    print('1')
