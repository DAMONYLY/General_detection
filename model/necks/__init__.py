from .fpn import FPN

def build_neck(cfg, channel_in = [1024, 512, 256]):
    """
    Arguments:
        name (str): the fpn types, [yolo_fpn, fpn]
        strides (list or None): the backbone out strides, like [32, 16, 8] (optional)
    Returns:
        FPN (module)
    """
    if cfg.name == 'fpn':
        channel_out = cfg.get('channel_out', 256)
        neck = FPN(channel_in, channel_out)
        neck.init_weights()
    else:
        raise NotImplementedError
    
    return neck