import numpy as np
import torch
import torch.nn as nn

def iou_xyxy_torch(boxes1, boxes2):
    """
    [N, 4] with [M, 4] return [N, M]
    """

    boxes1_area = torch.prod(boxes1[:, 2:] - boxes1[:, :2], -1)
    boxes2_area = torch.prod(boxes2[:, 2:] - boxes2[:, :2], -1)

    # 计算出boxes1与boxes2相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    right_down = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(left_up))
    inter_area = torch.prod(inter_section, -1)

    union_area = boxes1_area[:, None] + boxes2_area[None, :] - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

class Focal_Loss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="none"):
        super(Focal_Loss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.__loss = nn.BCELoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        focal_weight = torch.where(torch.eq(target, 1.), 1. - input, input)
        loss *= self.__alpha * torch.pow(focal_weight, self.__gamma)

        return loss

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.FC_loss = Focal_Loss()
        
        

    def forward(self, classifications, regressions, anchors, annotations):
 
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]


        l_smooth_loss = nn.SmoothL1Loss(reduction='none')
        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)


            IoU = iou_xyxy_torch(anchors[0, :, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # compute the loss for classification

            neg_mask = IoU_max < 0.4
            pos_mask = IoU_max > 0.5

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            pos_cls_preds = classification[pos_mask]
            pos_cls_targets = torch.zeros_like(pos_cls_preds)
            pos_cls_targets[torch.arange(pos_cls_targets.shape[0]), assigned_annotations[pos_mask, 4].long()] = 1

            neg_cls_preds = classification[neg_mask]
            neg_cls_targets = torch.zeros_like(neg_cls_preds)

            cls_preds = torch.cat([pos_cls_preds, neg_cls_preds], dim=0)
            cls_targets = torch.cat([pos_cls_targets, neg_cls_targets], dim=0)


            num_positive_anchors = pos_mask.sum()
            
            cls_loss = self.FC_loss(cls_preds, cls_targets)


            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if num_positive_anchors > 0:

                reg_targets = self.encode(anchor[pos_mask], assigned_annotations[pos_mask])
                reg_preds = regression[pos_mask]

                if torch.cuda.is_available():
                    reg_targets = reg_targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    reg_targets = reg_targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                regression_loss = l_smooth_loss(reg_preds, reg_targets)

                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

    def encode(self, anchor, gt_bbox, eps = 1e-6):
        
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

