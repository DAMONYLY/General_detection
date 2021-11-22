from copy import deepcopy

import torch
import torch.nn as nn
from thop import profile

def get_model_info(model, tsize):

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize * tsize / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}GFLOPs".format(params, flops)
    return info