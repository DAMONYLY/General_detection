import torch.nn as nn
import torch

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


    def forward(self, anchors, targets, regressions, classifications):
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
        
        cls_preds_assign = []
        reg_preds_assign = []
        cls_targets_assign = []
        reg_targets_assign = []

        batch_size = targets.shape[0]
        anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1)
        
        for batch_id in range(batch_size):
            classification = classifications[batch_id]
            regression = regressions[batch_id][..., :4]
            anchor = anchors[batch_id]
            target = targets[batch_id]
            target = target[target[:, 4] != -1]

            if target.shape[0] == 0:
                # print(target.shape)
                continue
            # compute IOU between anchor and target
            overlaps = iou_xyxy_torch(anchor, target[..., :4]) # [N, M]
            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            neg_mask = max_overlaps < self.neg_iou_thr
            pos_mask = max_overlaps > self.pos_iou_thr
            
            assign_targets = target[argmax_overlaps]
         
            # bulid cls targets and preds
            pos_cls_preds = classification[pos_mask]
            pos_cls_targets = torch.zeros_like(pos_cls_preds)
            
            neg_cls_preds = classification[neg_mask]
            neg_cls_targets = torch.zeros_like(neg_cls_preds)
            pos_cls_targets[torch.arange(pos_cls_targets.shape[0]), assign_targets[pos_mask, 4].long()] = 1

            # bulid reg targets and preds
            reg_preds = regression[pos_mask]
            neg_reg_preds = regression[neg_mask]

            if self.IOU_loss:
                reg_preds = yolo_decode(reg_preds, anchor[pos_mask])
                reg_preds = IOU_xyxy_torch_same(reg_preds, assign_targets[pos_mask, :4])
                reg_targets = torch.ones_like(reg_preds)
                pos_cls_targets[torch.arange(pos_cls_targets.shape[0]), assign_targets[pos_mask, 4].long()] = reg_preds.detach()
                
                neg_reg_preds = yolo_decode(neg_reg_preds, anchor[neg_mask])
                neg_reg_preds = IOU_xyxy_torch_same(neg_reg_preds, assign_targets[neg_mask, :4])
                neg_cls_targets[torch.arange(neg_cls_targets.shape[0]), assign_targets[neg_mask, 4].long()] = neg_reg_preds.detach()
            else:
                reg_targets = bbox2delta(anchor[pos_mask], assign_targets[pos_mask])
            
            cls_preds = torch.cat([pos_cls_preds, neg_cls_preds], dim = 0)
            cls_targets = torch.cat([pos_cls_targets, neg_cls_targets], dim = 0)
            
            cls_preds_assign.append(cls_preds)
            reg_preds_assign.append(reg_preds)

            cls_targets_assign.append(cls_targets)
            reg_targets_assign.append(reg_targets)


        cls_preds_assign = torch.cat(cls_preds_assign)
        reg_preds_assign = torch.cat(reg_preds_assign)

        cls_targets_assign = torch.cat(cls_targets_assign)
        reg_targets_assign = torch.cat(reg_targets_assign)
        # print('pos:', reg_preds_assign.shape[0], 
        #       'neg:', cls_preds_assign.shape[0] - reg_preds_assign.shape[0],
        #       'ignore:', anchors.shape[0] * anchors.shape[1] - cls_preds_assign.shape[0]) 

        return cls_preds_assign, reg_preds_assign, cls_targets_assign, reg_targets_assign
        

    