import torch

def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes



def bbox2delta(anchor, gt_bbox, eps = 1e-6):
    
    anchor_widths = anchor[..., 2] - anchor[..., 0]
    anchor_heights = anchor[..., 3] - anchor[..., 1]

    anchor_ctr_x = anchor[..., 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchor[..., 1] + 0.5 * anchor_heights


    gt_widths = gt_bbox[..., 2] - gt_bbox[..., 0]
    gt_heights = gt_bbox[..., 3] - gt_bbox[..., 1]
    gt_widths = torch.clamp(gt_widths, min=1)
    gt_heights = torch.clamp(gt_heights, min=1)
    gt_ctr_x = gt_bbox[..., 0] + 0.5 * gt_widths
    gt_ctr_y = gt_bbox[..., 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights


    targets_dw = torch.log(gt_widths / anchor_widths)
    targets_dh = torch.log(gt_heights / anchor_heights)

    reg_targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh), dim=-1
    )
    return reg_targets


def delta2bbox(feature, anchor):
    """
    Args:
        feature (torch.tensor): output of fpn. [B, num_per_anchor, W, H, reg].
        anchor (torch.tensor): anchors [num_per_grid, feature_w * feature_h, 4]

    output:
        feature (torch.tensor): after decode. [all_num_anchors, reg], in xyxy form.
    """
    anchor_widths  = anchor[..., 2] - anchor[..., 0]
    anchor_heights = anchor[..., 3] - anchor[..., 1]
    anchor_ctr_x   = anchor[..., 0] + 0.5 * anchor_widths
    anchor_ctr_y   = anchor[..., 1] + 0.5 * anchor_heights

    pred_dx = anchor_ctr_x + feature[..., 0] * anchor_widths
    pred_dy = anchor_ctr_y + feature[..., 1] * anchor_heights
    pred_dw = torch.exp(feature[..., 2]) * anchor_widths
    pred_dh = torch.exp(feature[..., 3]) * anchor_heights

    pred_x1 = pred_dx - 0.5*pred_dw
    pred_y1 = pred_dy - 0.5*pred_dh
    pred_x2 = pred_dx + 0.5*pred_dw
    pred_y2 = pred_dy + 0.5*pred_dh
    pred_reg = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim = -1).view(-1, 4)

    return pred_reg

def clip_bboxes(boxes, image_size):
    """
    Args:
        boxes (torch.tensor): output of decoded pred. [all_num_pred, reg+obj+cls].
        images (torch.tensor): input images

    output:
        boxes (torch.tensor): after clip. [all_num_pred, reg+obj+cls], in xyxy form.
    """
    height, width = image_size

    boxes[:, 0] = torch.clamp(boxes[:, 0], min=0)
    boxes[:, 1] = torch.clamp(boxes[:, 1], min=0)

    boxes[:, 2] = torch.clamp(boxes[:, 2], max=width)
    boxes[:, 3] = torch.clamp(boxes[:, 3], max=height)

    return boxes

