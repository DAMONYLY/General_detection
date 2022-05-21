from copy import deepcopy
import torch
from thop import profile

def get_model_info(model, tsize):

    if isinstance(tsize, list):
        tsize = tsize[0]
    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize * tsize / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}GFLOPs".format(params, flops)
    return info

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)