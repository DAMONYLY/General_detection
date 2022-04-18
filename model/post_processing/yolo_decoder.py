import torch
import config.cfg_example as cfg
from torchvision.ops import nms
strides = [32, 16, 8]

def yolo_decode(feature, anchor):
    """
    Args:
        feature (torch.tensor): output of fpn. [B, num_per_anchor, W, H, reg].
        anchor (torch.tensor): anchors [num_per_grid, feature_w * feature_h, 4]

    output:
        feature (torch.tensor): after decode. [all_num_anchors, reg], in xyxy form.
    """
    num_anchor, out_dim = feature.shape
    dtype = feature.dtype
    device = feature.device
    # anchor = anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1).type(dtype).to(device)
    # feature = feature.view(batch_size, num_anchor, w * h, out_dim)

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
    pred_reg = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim = -1).view(-1, out_dim)

    return pred_reg

def clip_bboxes(boxes, images):
    """
    Args:
        boxes (torch.tensor): output of decoded pred. [all_num_pred, reg+obj+cls].
        images (torch.tensor): input images

    output:
        boxes (torch.tensor): after clip. [all_num_pred, reg+obj+cls], in xyxy form.
    """
    batch_size, num_channel, height, width = images.shape

    boxes[:, 0] = torch.clamp(boxes[:, 0], min=0)
    boxes[:, 1] = torch.clamp(boxes[:, 1], min=0)

    boxes[:, 2] = torch.clamp(boxes[:, 2], max=width)
    boxes[:, 3] = torch.clamp(boxes[:, 3], max=height)

    return boxes

def nms_boxes(regression, classification, nms_score = 0.5):
    """
    Args:
        classification (torch.tensor)[all_num_pred, cls]
        regression (torch.tensor)[all_num_pred, reg+obj]
        nms_score (float)

    Returns:

    """
    finalResult = [[], [], []]

    finalScores = torch.Tensor([])
    finalAnchorBoxesIndexes = torch.Tensor([]).long()
    finalAnchorBoxesCoordinates = torch.Tensor([])

    if torch.cuda.is_available():
        finalScores = finalScores.cuda()
        finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
        finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

    for i in range(classification.shape[-1]):
        scores = torch.squeeze(classification[:, i])
        scores_over_thresh = (scores > 0.05)
        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just continue
            continue

        scores = scores[scores_over_thresh]
        anchorBoxes = torch.squeeze(regression)
        anchorBoxes = anchorBoxes[scores_over_thresh]
        anchors_nms_idx = nms(anchorBoxes, scores, nms_score)

        finalResult[0].extend(scores[anchors_nms_idx])
        finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
        finalResult[2].extend(anchorBoxes[anchors_nms_idx])

        finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
        finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
        if torch.cuda.is_available():
            finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

        finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
        finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

    return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]