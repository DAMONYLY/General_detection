import torch
import config.cfg_example as cfg
strides = [32, 16, 8]

def yolo_decode(feature, id):
    """
    Args:
        feature (torch.tensor): output of fpn. [B, per_anchor, W, H, out_dim].

    output:
        feature (torch.tensor): after decode. [all_num_anchors, out_dim].
    """
    batch_size, num_anchor, w, h, out_dim = feature.shape

    device = feature.device
    dtype = feature.dtype
    stride = strides[id]
    anchors = torch.tensor(cfg.MODEL['ANCHORS']).type(dtype).to(device)
    anchors = anchors[id].view(1, num_anchor, 1, 1, 2)

    conv_raw_dxdy = feature[..., 0:2]
    conv_raw_dwdh = feature[..., 2:4]
    conv_raw_conf = feature[..., 4:5]
    conv_raw_prob = feature[..., 5:]

    y = torch.arange(0, w).unsqueeze(1).repeat(1, h)
    x = torch.arange(0, h).unsqueeze(0).repeat(w, 1)
    grid_xy = torch.stack([x, y], dim=-1).view(1, 1, w, h, 2).repeat(batch_size, num_anchor, 1, 1, 1).type(dtype).to(device)

    pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
    pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
    # pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
    pred_conf = torch.sigmoid(conv_raw_conf)
    pred_prob = torch.sigmoid(conv_raw_prob)
    pred_bbox = torch.cat([pred_xy, pred_wh, pred_conf, pred_prob], dim=-1).view(-1, out_dim)

    return pred_bbox