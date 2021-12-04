'''
2021年11月29日16:25:10

用于产生anchor
'''


import numpy as np
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
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            # self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
            self.scales = np.array([2 ** 0])
    def make_base_anchor(self, base_size=16, ratios=None, scales=None):
        base_anchors = []
        for ratio in ratios:
            for scale in scales:
                base_anchors.extend([-scale * math.sqrt(ratio), -scale / math.sqrt(ratio), scale * math.sqrt(ratio), scale / math.sqrt(ratio)])
        base_anchors = (torch.tensor(base_anchors) * (base_size / 2)).reshape(-1, 4)
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
        all_anchors = grids[:, None, :] + anchors[None, :, :]
        all_anchors = all_anchors.reshape(-1, 4)
        return all_anchors

    def forward(self, image):
        """
        Args:
            image (torch.Tensor(B, C, W, H)): the image need to process


        Returns:
            anchors (torch.Tensor(X, 4)): anchor based on input image size with xyxy (bottom left, top right)
            X = (size/stride1)**2 *num_anchor + (size/stride2)**2 *num_anchor + (size/stride3)**2 *num_anchor
            large + medium + small
        """
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [image_shape // s for s in self.strides]
        dtype = image.dtype
        device = image.device
        all_anchors = []

        for idx, level in enumerate(self.pyramid_levels):

            base_anchors = self.make_base_anchor(self.sizes[idx], self.ratios, self.scales)
            one_anchors = self.generate_anchor_single_level(image_shapes[idx], self.strides[idx], base_anchors)
            all_anchors.append(one_anchors)

        all_anchors = torch.cat((all_anchors), dim=0)
        return all_anchors.type(dtype).to(device)

if __name__ == "__main__":
    gen = Anchors()
    in_img = torch.randn(12, 3, 416, 416)
    a = gen(in_img)
