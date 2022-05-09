import torch.nn as nn
import torch

from utils.tools import *
from ..post_processing.yolo_decoder import yolo_decode



class Max_iou_assigner(nn.Module):
    def __init__(self, cfg, pos_iou_thr, neg_iou_thr) -> None:
        super(Max_iou_assigner, self).__init__()

        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.IOU_loss = True if 'iou' in cfg.Model.loss.reg_loss.name.lower() else False


    def forward(self, anchors, targets, regressions, classifications):
        # TODO
        """
        Arguments:
            anchors : [[levels, N, w, h, 4],...] xyxy type
            target: [B, Max, 6] 6 contains [x,y,x,y,ind,mixind]
            classifications: [[B,N,w,h,20],...] model cls pred, 20指类别
            regressions: [[B, N,w,h,5],...] model reg pred
        Returns:
            cls_preds_assign (torch.tensor, [fpn_level*M, 20]): 模型预测的类别
            reg_preds_assign (torch.tensor, [fpn_level*M, 5]): 模型预测的坐标
            cls_targets_assign (torch.tensor, [fpn_level*M, 20]): 类别对应的标签
            reg_targets_assign (torch.tensor, [fpn_level*M, 5]): 坐标对应的标签
        """
        
        cls_preds_assign = []
        reg_preds_assign = []

        cls_targets_assign = []
        reg_targets_assign = []


        batch_size = targets.shape[0]

        device = targets.device
        dtype = targets.dtype

        anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1)
        
        for batch_id in range(batch_size):
            classification = classifications[batch_id]
            regression = regressions[batch_id][..., :4]
            anchor = anchors[batch_id].type(dtype).to(device)
            target = targets[batch_id]
            target = target[target[:, 4] != -1]

            if target.shape[0] == 0:
                # print(target.shape)
                continue
            # 计算IOU between anchor and target
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
                reg_targets = self.encode(anchor[pos_mask], assign_targets[pos_mask])
            
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


        return cls_preds_assign, reg_preds_assign, cls_targets_assign, reg_targets_assign
        
    def anchor_featuremap(self, anchor, shape_w, shape_h):
        """
        anchor [num_per_grid, 2], wh

        Return:
                [num_per_grid, shape_w, shape_h, 4], xyxy
        """
        num_per_grid = anchor.shape[0]
        y = (torch.arange(0, shape_w).unsqueeze(1) + 0.5).repeat(1, shape_h)
        x = (torch.arange(0, shape_h).unsqueeze(0) + 0.5).repeat(shape_w, 1)
        # [num_per_grid, shape_w, shape_h, x_c y_c]
        grid_xy = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(num_per_grid, 1, 1, 1).type(anchor.dtype).to(anchor.device)
        anchor = anchor.view(num_per_grid, 1, 1, 2)
        # [num_per_grid, shape_w, shape_h, (x_c y_c w h)]
        anchor = torch.cat((grid_xy, anchor), dim = -1)
        return anchor


    def convert_wh_xyxy(self, anchor):
        anchor = torch.tensor(anchor)
        widths = anchor[..., 0]
        heights = anchor[..., 1]
        output = torch.stack([-widths/2, -heights/2, widths/2, heights/2], dim=-1)
        return output

    def get_center(self, target):
        widths = target[..., 2] + target[..., 0]
        heights = target[..., 3] + target[..., 1]
        output = torch.stack([widths/2, heights/2], dim=-1)
        return output

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

        # targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        # targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
        targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights


        targets_dw = torch.log(gt_widths / anchor_widths)
        targets_dh = torch.log(gt_heights / anchor_heights)

        reg_targets = torch.stack(
            (targets_dx, targets_dy, targets_dw, targets_dh), dim=-1
        )
        return reg_targets