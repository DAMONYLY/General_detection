import torch
import config.cfg_example as cfg
strides = [32, 16, 8]

def yolo_decode(feature, anchor):
    """
    Args:
        feature (torch.tensor): output of fpn. [B, num_per_anchor, W, H, out_dim].
        anchor (torch.tensor): anchors [num_per_grid, feature_w * feature_h, 4]

    output:
        feature (torch.tensor): after decode. [all_num_anchors, out_dim].
    """
    batch_size, num_anchor, w, h, out_dim = feature.shape
    dtype = feature.dtype
    device = feature.device
    anchor = anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1).type(dtype).to(device)
    feature = feature.view(batch_size, num_anchor, w * h, out_dim)

    anchor_widths  = anchor[..., 2] - anchor[..., 0]
    anchor_heights = anchor[..., 3] - anchor[..., 1]
    anchor_ctr_x   = anchor[..., 0] + 0.5 * anchor_widths
    anchor_ctr_y   = anchor[..., 1] + 0.5 * anchor_heights

    pred_dx = anchor_ctr_x + feature[..., 0] * anchor_widths
    pred_dy = anchor_ctr_y + feature[..., 1] * anchor_heights
    pred_dw = torch.exp(feature[..., 2]) * anchor_widths
    pred_dh = torch.exp(feature[..., 3]) * anchor_heights

    pred_reg = torch.stack([pred_dx, pred_dy, pred_dw, pred_dh], dim = -1)
    pred_obj = torch.sigmoid(feature[..., 4:5])
    pred_cls = torch.sigmoid(feature[..., 5:])

    pred_bbox = torch.cat([pred_reg, pred_obj, pred_cls], dim=-1).view(-1, out_dim)

    return pred_bbox