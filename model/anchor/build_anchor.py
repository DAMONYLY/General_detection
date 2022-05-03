import numpy as np
import torch
import torch.nn as nn

class Anchors(nn.Module):
    def __init__(self, cfg):
        super(Anchors, self).__init__()

        self.strides = cfg.strides
        if isinstance(cfg.sizes, int):
            self.sizes = torch.tensor([cfg.sizes for _ in self.strides])
        else:
            self.sizes = cfg.sizes
        self.ratios = torch.tensor(cfg.ratios)
        scales = cfg.scales
        if isinstance(scales[0], str):
            self.scales = torch.tensor([eval(scale) for scale in scales])
        else:
            self.scales = scales
        # if cfg.ratios is None:
        #     self.ratios = np.array([0.5, 1, 2])
        # if cfg.scales is None:
        #     self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        self.num_anchors = len(self.ratios) * len(self.scales)        
        
    def forward(self, image):
        #TODO 可以传入fpn层的out结果的shape来生成anchor，这样就不需要事先定义pyramid_levels
        # strides 可以自己定义， size也预先定义， ratio，scales也一样。
        # 检查，需要strides和fpn层结果层数一致，值对不对可能需要img来检查
        # assert len(features) == len(self.strides), \
        # 'the number of features should be equal to number of stides, but got '\
        #     f'{features} and {self.strides}' 
        
        device = image.device
        dtype = image.dtype
        image_shape = torch.tensor(image.size()[-2:])
        feature_shapes = [torch.div(image_shape+stride-1, stride, rounding_mode = 'trunc') 
                                    for stride in self.strides]

        # feature_shapes = [feature.size()[-2:] for feature in features]

        # compute anchors over all pyramid levels
        all_anchors = []

        for idx, p in enumerate(feature_shapes):
            base_anchors = self.generate_base_anchors(base_size=self.sizes[idx], ratio=self.ratios, scale=self.scales)
            grid_anchors = self.grid_anchors(feature_shapes[idx], self.strides[idx], base_anchors)
            all_anchors.append(grid_anchors)

        all_anchors = torch.cat(all_anchors).type(dtype).to(device)

        return all_anchors
    
    def generate_base_anchors(self, base_size, ratio, scale):

        w = (base_size * 1/torch.sqrt(ratio)[:, None] * scale[None, :]).view(-1)
        h = (base_size * torch.sqrt(ratio)[:, None] * scale[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            - 0.5 * w, - 0.5 * h, 0.5 * w, 0.5 * h
            ]
        base_anchors = torch.stack(base_anchors, dim=-1) # [num_anchors, 4]
        
        return base_anchors

    def grid_anchors(self, feature_shape, stride, base_anchors):
        h_shape, w_shape = feature_shape
        grid_x = torch.arange(0, w_shape) * stride
        grid_y = torch.arange(0, h_shape) * stride 
        
        shift_x, shift_y = self.meshgrid(grid_x, grid_y)
        shift_anchors = torch.stack((shift_x.flatten(), shift_y.flatten(),
                                     shift_x.flatten(), shift_y.flatten()), dim=-1)
        
        grid_anchors = base_anchors[None, :, :] + shift_anchors[:, None, :]
        grid_anchors = grid_anchors.view(-1, 4)

        return grid_anchors
    
    def meshgrid(self, x, y):
        # x;[136] y[100]
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)          
        return xx, yy
# class Anchors(nn.Module):
    # def __init__(self, cfg, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
#         super(Anchors, self).__init__()

#         # if pyramid_levels is None:
#         self.pyramid_levels = [3, 4, 5, 6, 7]
#         # if strides is None:
#         self.strides = [2 ** x for x in self.pyramid_levels]
#         # if sizes is None:
#         self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
#         # if ratios is None:
#         self.ratios = np.array([0.5, 1, 2])
#         # if scales is None:
#         # self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
#         self.scales = np.array([2 ** 0])

#     def forward(self, image):
#         #TODO 可以传入fpn层的out结果的shape来生成anchor，这样就不需要事先定义pyramid_levels
#         # strides 可以自己定义， size也预先定义， ratio，scales也一样。
#         # 检查，需要strides和fpn层结果层数一致，值对不对可能需要img来检查
#         device = image.device
#         dtype = image.dtype
#         image_shape = image.shape[2:]
#         image_shape = np.array(image_shape)
#         image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

#         # compute anchors over all pyramid levels
#         all_anchors = np.zeros((0, 4)).astype(np.float32)

#         for idx, p in enumerate(self.pyramid_levels):
#             anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
#             shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
#             all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

#         all_anchors = np.expand_dims(all_anchors, axis=0)
#         all_anchors = torch.from_numpy(all_anchors).squeeze().type(dtype).to(device)
#         return all_anchors

# def generate_anchors(base_size=16, ratios=None, scales=None):
#     """
#     Generate anchor (reference) windows by enumerating aspect ratios X
#     scales w.r.t. a reference window.
#     """

#     if ratios is None:
#         ratios = np.array([0.5, 1, 2])

#     if scales is None:
#         scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

#     num_anchors = len(ratios) * len(scales)

#     # initialize output anchors
#     anchors = np.zeros((num_anchors, 4))

#     # scale base_size
#     anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

#     # compute areas of anchors
#     areas = anchors[:, 2] * anchors[:, 3]

#     # correct for ratios
#     anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
#     anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

#     # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
#     anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
#     anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

#     return anchors

# def compute_shape(image_shape, pyramid_levels):
#     """Compute shapes based on pyramid levels.

#     :param image_shape:
#     :param pyramid_levels:
#     :return:
#     """
#     image_shape = np.array(image_shape[:2])
#     image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
#     return image_shapes


# def shift(shape, stride, anchors):
#     shift_x = np.arange(0, shape[1]) * stride
#     shift_y = np.arange(0, shape[0]) * stride

#     shift_x, shift_y = np.meshgrid(shift_x, shift_y)

#     shifts = np.vstack((
#         shift_x.ravel(), shift_y.ravel(),
#         shift_x.ravel(), shift_y.ravel()
#     )).transpose()

#     # add A anchors (1, A, 4) to
#     # cell K shifts (K, 1, 4) to get
#     # shift anchors (K, A, 4)
#     # reshape to (K*A, 4) shifted anchors
#     A = anchors.shape[0]
#     K = shifts.shape[0]
#     all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
#     all_anchors = all_anchors.reshape((K * A, 4))

#     return all_anchors


