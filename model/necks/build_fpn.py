import torch.nn as nn
from model.necks.yolo_fpn import FPN_YOLOV3
from model.necks.fpn import FPN

def build_fpn(name, strides = None, channel_in = [1024, 512, 256]):
    """
    Arguments:
        name (str): the fpn types, [yolo_fpn, fpn]
        strides (list or None): the backbone out strides, like [32, 16, 8] (optional)
    Returns:
        FPN (module)
    """
    if name == 'yolo_fpn':
        return FPN_YOLOV3()
    elif name == 'fpn':
        channel_out = 256
        return FPN(strides, channel_in, channel_out)
    else:
        raise NotImplementedError