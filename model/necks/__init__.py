
from .yolo_fpn import FPN_YOLOV3
from .fpn import FPN

def build_fpn(cfg, channel_in = [1024, 512, 256]):
    """
    Arguments:
        name (str): the fpn types, [yolo_fpn, fpn]
        strides (list or None): the backbone out strides, like [32, 16, 8] (optional)
    Returns:
        FPN (module)
    """
    if cfg.name == 'yolo_fpn':
        return FPN_YOLOV3()
    elif cfg.name == 'fpn':
        channel_out = cfg.get('channel_out', 256)
        return FPN(channel_in, channel_out)
    else:
        raise NotImplementedError