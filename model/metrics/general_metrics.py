'''
2021年11月29日17:00:15
通用label assign
yolo形式
'''
import torch.nn as nn
import torch
import sys
sys.path.append('/raid/yaoliangyong/General_detection/')
from utils.tools import *



class label_assign(nn.Module):
    def __init__(self, cfg, metrics_type, pos_iou_thr = 0.3, neg_iou_thr = 0.3) -> None:
        super(label_assign, self).__init__()
        self.metrics = metrics_type
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.strides = [32, 16, 8]
        self.anchor = self.convert_wh_xyxy(cfg.MODEL['ANCHORS'])


    def forward(self, q_anchors, targets, regressions, classifications):
        # TODO
        """
        Arguments:
            anchors : [levels, N,4] xyxy
            target: [M, 7] 6 contains [x,y,x,y,ind,mixind,imgh_batch_id]
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
        obj_preds_assign = []
        cls_targets_assign = []
        reg_targets_assign = []
        obj_targets_assign = []

        num, _ = targets.shape
        batch_size = int(targets[-1,-1] + 1)
        device = targets.device
        dtype = targets.dtype
        num_idxs = torch.arange(num)
        targets_batch_ids = targets[..., -1].long()
        levels = len(classifications)
        # anchors = q_anchors.unsqueeze(0).repeat(batch_size, 1, 1, 1).type(dtype).to(device)
        anchors = self.anchor.type(dtype).to(device)

        for level in range(levels):
            classification = classifications[level]
            regression, objecteness = regressions[level][..., :4], regressions[level][..., -1:]
            bbox_annotation = self.convert_to_origin(targets)/self.strides[level] # [M, 4]
            target_center = self.get_center(targets)/self.strides[level] # [M, 2]
            target_grids = target_center.long()
            anchor = anchors[level, ...]
            # 计算IOU between anchor and target
            # overlaps = iou_xyxy_torch_batch(anchor, bbox_annotation)
            overlaps = iou_xyxy_torch(anchor, bbox_annotation) # [N, M]
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            overlaps_mask = max_overlaps > self.pos_iou_thr

            targets_batch_ids = targets_batch_ids[overlaps_mask]
            argmax_overlaps = argmax_overlaps[overlaps_mask]
            target_grids = target_grids[overlaps_mask]
            targets = targets[overlaps_mask]
            num_idxs = torch.arange(targets.shape[0])

            assign_anchors = anchor[argmax_overlaps] # [M, 4] 
            obj_targets = torch.zeros_like(objecteness)
            # Get the model classification and regression of the grid corresponding to the target center
            # cls_preds: [M, classification_out]. reg_preds: [M, regression_out]
            cls_preds = classification[targets_batch_ids, argmax_overlaps, target_grids[..., 0], target_grids[..., 1]]
            reg_preds = regression[targets_batch_ids, argmax_overlaps, target_grids[..., 0], target_grids[..., 1]]
            
            # cls_preds = cls_preds[overlaps_mask]
            # reg_preds = reg_preds[overlaps_mask]
            # assign_anchors = assign_anchors[overlaps_mask]
            # Obtain the  regression targets. [M, regression_out]
            reg_targets = self.encode(assign_anchors, targets/self.strides[level])
            # Obtain the  classification targets. [M, classification_out]
            cls_targets = torch.zeros_like(cls_preds)
            cls_targets[num_idxs, targets[..., 4].long()] = 1


            obj_targets = torch.zeros_like(objecteness)
            obj_targets[targets_batch_ids, argmax_overlaps, target_grids[..., 0], target_grids[..., 1]] = 1
            
            cls_preds_assign.append(cls_preds)
            reg_preds_assign.append(reg_preds)
            obj_preds_assign.append(objecteness.contiguous().view(-1, 1))
            cls_targets_assign.append(cls_targets)
            reg_targets_assign.append(reg_targets)
            obj_targets_assign.append(obj_targets.contiguous().view(-1, 1))

        cls_preds_assign = torch.cat(cls_preds_assign)
        reg_preds_assign = torch.cat(reg_preds_assign)
        obj_preds_assign = torch.cat(obj_preds_assign)
        cls_targets_assign = torch.cat(cls_targets_assign)
        reg_targets_assign = torch.cat(reg_targets_assign)
        obj_targets_assign = torch.cat(obj_targets_assign)

        return cls_preds_assign, reg_preds_assign, obj_preds_assign, cls_targets_assign, reg_targets_assign, obj_targets_assign


    def convert_to_origin(self, target):
        """
        Convert from original map coordinates to origin coordinates.
        Args:
            target (torch.Tensors): [M, 7]
        Returns:
            target (torch.Tensors): [M, 4]
        """
        widths = target[..., 2] - target[..., 0]
        heights = target[..., 3] - target[..., 1]
        output = torch.stack([-widths/2, -heights/2, widths/2, heights/2], dim=-1)
        return output
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

    def encode(self, ex_rois, gt_rois):
        dtype = gt_rois.dtype
        # change [xmin, ymin, xmax, ymax] to [x_center, y_center, width, height]
        ex_widths = ex_rois[..., 2] - ex_rois[..., 0]
        ex_heights = ex_rois[..., 3] - ex_rois[..., 1]
        ex_widths = torch.clamp(ex_widths, min=1)
        ex_heights = torch.clamp(ex_heights, min=1)
        ex_ctr_x = ex_rois[..., 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[..., 1] + 0.5 * ex_heights

        gt_widths = gt_rois[..., 2] - gt_rois[..., 0]
        gt_heights = gt_rois[..., 3] - gt_rois[..., 1]
        gt_widths = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)
        gt_ctr_x = gt_rois[..., 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[..., 1] + 0.5 * gt_heights

        # targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        # targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dx = gt_ctr_x - gt_ctr_x.to(torch.int16).to(dtype)
        targets_dy = gt_ctr_y - gt_ctr_y.to(torch.int16).to(dtype)


        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)

        reg_targets = torch.stack(
            (targets_dx, targets_dy, targets_dw, targets_dh), dim=-1
        )
        return reg_targets
from torch.utils.data import Dataset, DataLoader
from utils.datasets import VocDataset
if __name__ == "__main__":

    voc_dataset = VocDataset(anno_file_type="train", img_size=448)
    dataloader = DataLoader(voc_dataset, shuffle=True, batch_size=1, num_workers=0)
    test = label_assign('1',0.1,0.2)

    for i, (img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(dataloader):
        if i==0:
            print(img.shape)
            print(label_sbbox.shape)
            print(label_mbbox.shape)
            print(label_lbbox.shape)
            print(sbboxes.shape)
            print(mbboxes.shape)
            print(lbboxes.shape)
        anchors = torch.rand_like(sbboxes)
        targets = torch.rand((1,150,5))[:, :10]
        targets[:, :, -1] *= 10 
        # label_sbbox = label_sbbox.view(1, 56, 56, -1)
        out = test.forward(anchors.squeeze(0), targets, 1, 1)