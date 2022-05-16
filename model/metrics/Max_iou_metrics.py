import torch.nn as nn
import torch
from model.metrics.assign_results import AssignResult

from utils.tools import *
from ..post_processing.yolo_decoder import yolo_decode
from model.utils import bbox2delta



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
        self.IOU_loss = True if 'iou' in cfg.Model.loss.reg_loss.name.lower() else False
        self.num_classes = cfg.Classes.num
        
    def forward(self, bboxes, targets):
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
        if not targets.size(0):
            return
        target = targets[targets[:, 4] != -1]
        gt_bboxes = target[:, :4]
        gt_labels = target[:, 4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute IOU between anchor and target
        overlaps = iou_xyxy_torch(bboxes, gt_bboxes) # [N, M]
        max_overlaps, argmax_overlaps = overlaps.max(dim=1) # [N]

        assigned_gt_inds = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long) # [N]
        
        neg_mask = max_overlaps < self.neg_iou_thr
        pos_mask = max_overlaps > self.pos_iou_thr
        
        assigned_gt_inds[pos_mask] = argmax_overlaps[pos_mask] + 1
        assigned_gt_inds[neg_mask] = 0
        
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[pos_mask] = gt_labels[assigned_gt_inds[pos_mask] - 1].long()
        
        return AssignResult(num_gt, num_bboxes, assigned_gt_inds, max_overlaps, assigned_labels, bboxes, target)


    # def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
    #     assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
    #         gt_bboxes.shape[0] > self.gpu_assign_thr) else False
    #     # compute overlap and assign gt on CPU when number of GT is large
    #     if assign_on_cpu:
    #         device = bboxes.device
    #         bboxes = bboxes.cpu()
    #         gt_bboxes = gt_bboxes.cpu()
    #         if gt_bboxes_ignore is not None:
    #             gt_bboxes_ignore = gt_bboxes_ignore.cpu()
    #         if gt_labels is not None:
    #             gt_labels = gt_labels.cpu()

    #     overlaps = self.iou_calculator(gt_bboxes, bboxes)

    #     if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
    #             and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
    #         if self.ignore_wrt_candidates:
    #             ignore_overlaps = self.iou_calculator(
    #                 bboxes, gt_bboxes_ignore, mode='iof')
    #             ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
    #         else:
    #             ignore_overlaps = self.iou_calculator(
    #                 gt_bboxes_ignore, bboxes, mode='iof')
    #             ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
    #         overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

    #     assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
    #     if assign_on_cpu:
    #         assign_result.gt_inds = assign_result.gt_inds.to(device)
    #         assign_result.max_overlaps = assign_result.max_overlaps.to(device)
    #         if assign_result.labels is not None:
    #             assign_result.labels = assign_result.labels.to(device)
    #     return assign_result

    # def assign_wrt_overlaps(self, overlaps, gt_labels=None):
    #     """Assign w.r.t. the overlaps of bboxes with gts.

    #     Args:
    #         overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
    #             shape(k, n).
    #         gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

    #     Returns:
    #         :obj:`AssignResult`: The assign result.
    #     """
    #     num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

    #     # 1. assign -1 by default
    #     assigned_gt_inds = overlaps.new_full((num_bboxes, ),
    #                                          -1,
    #                                          dtype=torch.long)

    #     # for each anchor, which gt best overlaps with it
    #     # for each anchor, the max iou of all gts
    #     max_overlaps, argmax_overlaps = overlaps.max(dim=0)
    #     # for each gt, which anchor best overlaps with it
    #     # for each gt, the max iou of all proposals
    #     gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

    #     # 2. assign negative: below
    #     # the negative inds are set to be 0
    #     if isinstance(self.neg_iou_thr, float):
    #         assigned_gt_inds[(max_overlaps >= 0)
    #                          & (max_overlaps < self.neg_iou_thr)] = 0
    #     elif isinstance(self.neg_iou_thr, tuple):
    #         assert len(self.neg_iou_thr) == 2
    #         assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
    #                          & (max_overlaps < self.neg_iou_thr[1])] = 0

    #     # 3. assign positive: above positive IoU threshold
    #     pos_inds = max_overlaps >= self.pos_iou_thr
    #     assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

    #     if self.match_low_quality:
    #         # Low-quality matching will overwrite the assigned_gt_inds assigned
    #         # in Step 3. Thus, the assigned gt might not be the best one for
    #         # prediction.
    #         # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
    #         # bbox 1 will be assigned as the best target for bbox A in step 3.
    #         # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
    #         # assigned_gt_inds will be overwritten to be bbox 2.
    #         # This might be the reason that it is not used in ROI Heads.
    #         for i in range(num_gts):
    #             if gt_max_overlaps[i] >= self.min_pos_iou:
    #                 if self.gt_max_assign_all:
    #                     max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
    #                     assigned_gt_inds[max_iou_inds] = i + 1
    #                 else:
    #                     assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
    #     # 依据assigned_gt_inds把对应assign的label填上
    #     if gt_labels is not None:
    #         assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
    #         pos_inds = torch.nonzero(
    #             assigned_gt_inds > 0, as_tuple=False).squeeze()
    #         if pos_inds.numel() > 0:
    #             assigned_labels[pos_inds] = gt_labels[
    #                 assigned_gt_inds[pos_inds] - 1]
    #     else:
    #         assigned_labels = None
    #     # print('pos:',sum(assigned_gt_inds > 0),
    #     #       'neg:', sum(assigned_gt_inds == 0),
    #     #       'ignore:', sum(assigned_gt_inds == -1))
    #     return AssignResult(
    #         num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        


    # def forward(self, anchors, targets, regressions, classifications):
    #     """
    #     Arguments:
    #         anchors(tensor): Generated anchors, shape: [N, 4] in xyxy form.
    #         target(tensor): Groundtruth bboxes, shape: [B, _, 5],
    #                         5 contains (x, y, x, y, label), if label==-1, means no bboxes.
    #         regressions(tensor): Model reg pred, shape: [B, N, 4]
    #         classifications(tensor): Model cls pred, shape: [B, N, num_classes].
    #     Returns:
    #         cls_preds_assign(tensor), Model class prediction after label assign,
    #                                   shape: [M1, 20]
    #         reg_preds_assign(tensor), Model bbox prediction after label assign,
    #                                   shape: [M2, 4]
    #         cls_targets_assign(tensor), The labels corresponding to cls_preds_assign,
    #                                     shape: [M1, 20]
    #         reg_targets_assign(tensor), The labels corresponding to reg_preds_assign,
    #                                     shape: [M2, 4]
    #     """
        
    #     cls_preds_assign = []
    #     reg_preds_assign = []
    #     cls_targets_assign = []
    #     reg_targets_assign = []

    #     batch_size = targets.shape[0]
    #     anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1)
        
    #     for batch_id in range(batch_size):
    #         classification = classifications[batch_id]
    #         regression = regressions[batch_id][..., :4]
    #         anchor = anchors[batch_id]
    #         target = targets[batch_id]
    #         target = target[target[:, 4] != -1]

    #         if target.shape[0] == 0:
    #             # print(target.shape)
    #             continue
    #         # compute IOU between anchor and target
    #         overlaps = iou_xyxy_torch(anchor, target[..., :4]) # [N, M]
    #         max_overlaps, argmax_overlaps = overlaps.max(dim=1)

    #         neg_mask = max_overlaps < self.neg_iou_thr
    #         pos_mask = max_overlaps > self.pos_iou_thr
            
    #         assign_targets = target[argmax_overlaps]
         
    #         # bulid cls targets and preds
    #         pos_cls_preds = classification[pos_mask]
    #         pos_cls_targets = torch.zeros_like(pos_cls_preds)
            
    #         neg_cls_preds = classification[neg_mask]
    #         neg_cls_targets = torch.zeros_like(neg_cls_preds)
    #         pos_cls_targets[torch.arange(pos_cls_targets.shape[0]), assign_targets[pos_mask, 4].long()] = 1

    #         # bulid reg targets and preds
    #         reg_preds = regression[pos_mask]
    #         neg_reg_preds = regression[neg_mask]

    #         if self.IOU_loss:
    #             reg_preds = yolo_decode(reg_preds, anchor[pos_mask])
    #             reg_preds = IOU_xyxy_torch_same(reg_preds, assign_targets[pos_mask, :4])
    #             reg_targets = torch.ones_like(reg_preds)
    #             pos_cls_targets[torch.arange(pos_cls_targets.shape[0]), assign_targets[pos_mask, 4].long()] = reg_preds.detach()
                
    #             neg_reg_preds = yolo_decode(neg_reg_preds, anchor[neg_mask])
    #             neg_reg_preds = IOU_xyxy_torch_same(neg_reg_preds, assign_targets[neg_mask, :4])
    #             neg_cls_targets[torch.arange(neg_cls_targets.shape[0]), assign_targets[neg_mask, 4].long()] = neg_reg_preds.detach()
    #         else:
    #             reg_targets = bbox2delta(anchor[pos_mask], assign_targets[pos_mask])
            
    #         cls_preds = torch.cat([pos_cls_preds, neg_cls_preds], dim = 0)
    #         cls_targets = torch.cat([pos_cls_targets, neg_cls_targets], dim = 0)
            
    #         cls_preds_assign.append(cls_preds)
    #         reg_preds_assign.append(reg_preds)

    #         cls_targets_assign.append(cls_targets)
    #         reg_targets_assign.append(reg_targets)


    #     cls_preds_assign = torch.cat(cls_preds_assign)
    #     reg_preds_assign = torch.cat(reg_preds_assign)

    #     cls_targets_assign = torch.cat(cls_targets_assign)
    #     reg_targets_assign = torch.cat(reg_targets_assign)
    #     # print('pos:', reg_preds_assign.shape[0], 
    #     #       'neg:', cls_preds_assign.shape[0] - reg_preds_assign.shape[0],
    #     #       'ignore:', anchors.shape[0] * anchors.shape[1] - cls_preds_assign.shape[0]) 

    #     return cls_preds_assign, reg_preds_assign, cls_targets_assign, reg_targets_assign
        

    