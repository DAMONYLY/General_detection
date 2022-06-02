import torch.nn as nn
import torch

from .assign_results import AssignResult
from utils.tools import *




class Max_iou_assigner(nn.Module):
    """
    Assign GT to each anchor based on the maximum IOU.

    Args:
        cfg (CFNODE): yaml config 
        pos_iou_thr (float): Minimum IOU for positive samples
        neg_iou_thr (float): Maximum IOU for negative samples
    """
    
    def __init__(self, cfg, pos_iou_thr, neg_iou_thr) -> None:
        super(Max_iou_assigner, self).__init__()

        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.num_classes = cfg.Classes.num
        
    def assign(self, bboxes, targets, num_level_bboxes, **kwargs):
        """
        Arguments:
            anchors(tensor): Generated anchors, shape: [N, 4] in xyxy form.
            target(tensor): Groundtruth bboxes, shape: [B, _, 5],
                            5 contains (x, y, x, y, label), if label==-1, means no bboxes.
            regressions(tensor): Model reg pred, shape: [B, N, 4]
            classifications(tensor): Model cls pred, shape: [B, N, num_classes].
        Returns:
            cls_preds_assign(tensor), Model class prediction after label assign,
                                      shape: [M1, 20]
            reg_preds_assign(tensor), Model bbox prediction after label assign,
                                      shape: [M2, 4]
            cls_targets_assign(tensor), The labels corresponding to cls_preds_assign,
                                        shape: [M1, 20]
            reg_targets_assign(tensor), The labels corresponding to reg_preds_assign,
                                        shape: [M2, 4]
        """
        
        target = targets[targets[:, 4] != -1]
        gt_bboxes = target[:, :4]
        gt_labels = target[:, 4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        overlaps = iou_xyxy_torch(bboxes, gt_bboxes) # [N, M]
        assigned_gt_inds = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long) # [N]

        if num_gt == 0 or num_bboxes == 0:
            assigned_gt_inds[:] = 0 # [N]
            max_overlaps = overlaps.new_full((num_bboxes, ), 0)
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            return AssignResult(num_gt, num_bboxes, assigned_gt_inds, max_overlaps, assigned_labels, bboxes, target)
        # compute IOU between anchor and target

        max_overlaps, argmax_overlaps = overlaps.max(dim=1) # [N]
        max_gt_overlaps, argmax_gt_overlaps = overlaps.max(dim=0)
        
        neg_mask = max_overlaps < self.neg_iou_thr
        pos_mask = max_overlaps > self.pos_iou_thr
        
        assigned_gt_inds[pos_mask] = argmax_overlaps[pos_mask] + 1
        assigned_gt_inds[neg_mask] = 0
        
        for j in range(num_gt):
            index = (overlaps[:, j] == max_gt_overlaps[j])
            assigned_gt_inds[index] = j + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1].long()
        else:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)

        
        return AssignResult(num_gt, num_bboxes, assigned_gt_inds, max_overlaps, assigned_labels, bboxes, target)


    