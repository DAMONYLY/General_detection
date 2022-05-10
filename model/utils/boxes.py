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

