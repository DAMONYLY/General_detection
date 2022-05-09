import torch
import torch.nn as nn

class Anchors(nn.Module):
    def __init__(self, cfg):
        super(Anchors, self).__init__()

        self.strides = cfg.strides
        if isinstance(cfg.sizes, int):
            self.sizes = torch.tensor([cfg.sizes for _ in self.strides])
        else:
            self.sizes = torch.tensor(cfg.sizes)
        self.ratios = torch.tensor(cfg.ratios)
        scales = cfg.scales
        if isinstance(scales[0], str):
            self.scales = torch.tensor([eval(scale) for scale in scales])
        else:
            self.scales = torch.tensor(scales)
        self.num_anchors = len(self.ratios) * len(self.scales)        
        assert self.num_anchors == cfg.num, \
            'the number of anchors per grid should be equal to len(ratios) * len(scales),'\
            'but got ' f'{cfg.num} and {self.num_anchors}'
        
    def forward(self, image_size, device, dtype):
        
        image_shape = torch.tensor(image_size)
        feature_shapes = [torch.div(image_shape+stride-1, stride, rounding_mode = 'trunc') 
                                    for stride in self.strides]

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

        # use float anchor and the anchor's center is aligned with the pixel center
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
