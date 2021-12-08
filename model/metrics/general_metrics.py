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
        # self.anchor = self.convert_wh_xyxy(cfg.MODEL['ANCHORS'])


    def forward(self, q_anchors, targets, regressions, classifications):
        # TODO
        """
        Arguments:
            anchors : [levels, N,4] xyxy
            target: [B, M, 6] 6 contains xyxy,ind,mixind
            classifications: [[B,N,w,h,20],...] model cls pred, 20指类别
            regressions: [[B, N,w,h,5],...] model reg pred
        
            一个anchor 最多一个anchor， 一个GT至少有一个anchor
        """
        
        cls_preds_assign = []
        reg_preds_assign = []
        cls_targets_assign = []
        reg_targets_assign = []
        batch_size, num, _ = targets.shape
        device = targets.device
        dtype = targets.dtype
        levels = len(classifications)
        # anchors = self.anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1).type(dtype).to(device)
        anchors = q_anchors.unsqueeze(0).repeat(batch_size, 1, 1, 1).type(dtype).to(device)
        # classifications = torch.rand((1, 150, 20))
        for level in range(levels):
            classification = classifications[level]
            regression = regressions[level]
            bbox_annotation = self.convert_to_origin(targets)/self.strides[level]
            target_center = self.get_center(targets)/self.strides[level]
            target_grids = target_center.long()
            anchor = anchors[:, level, ...]
            cls_targets = torch.zeros_like
            # 计算IOU between anchor and target
            overlaps = iou_xyxy_torch_batch(anchor, bbox_annotation)
            
            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            batch_idxs = torch.arange(batch_size).reshape(-1, 1)
            num_idxs = torch.arange(num).reshape(1, -1)
            assign_anchors = anchor[batch_idxs, argmax_overlaps]
            cls_preds = classification[batch_idxs, argmax_overlaps, target_grids[..., 0], target_grids[..., 1]]
            reg_preds = regression[batch_idxs, argmax_overlaps, target_grids[..., 0], target_grids[..., 1]]

            reg_targets = self.encode(assign_anchors, targets)
            cls_targets = torch.zeros_like(cls_preds)
            cls_targets[batch_idxs, num_idxs, targets[..., 4].long()] = 1

            cls_preds_assign.append(cls_preds)
            reg_preds_assign.append(reg_preds)
            cls_targets_assign.append(cls_targets)
            reg_targets_assign.append(reg_targets)

        cls_preds_assign = torch.stack(cls_preds_assign)
        reg_preds_assign = torch.stack(reg_preds_assign)
        cls_targets_assign = torch.stack(cls_targets_assign)
        reg_targets_assign = torch.stack(reg_targets_assign)

        return cls_preds_assign, reg_preds_assign, cls_targets_assign, reg_targets_assign


    def convert_to_origin(self, target):
        """
        Convert from original map coordinates to origin coordinates.
        Args:
            target (torch.Tensors): [B, M, 4]
        Returns:
            target (torch.Tensors): [B, M, 4]
        """
        widths = target[..., 2] - target[..., 0]
        heights = target[..., 3] - target[..., 1]
        output = torch.stack([-widths/2, -heights/2, widths/2, heights/2], dim=2)
        return output
    def convert_wh_xyxy(self, anchor):
        anchor = torch.tensor(anchor)
        widths = anchor[..., 0]
        heights = anchor[..., 1]
        output = torch.stack([-widths/2, -heights/2, widths/2, heights/2], dim=-1)
        return output
    def get_center(self, target):
        widths = target[..., 2] - target[..., 0]
        heights = target[..., 3] - target[..., 1]
        output = torch.stack([widths/2, heights/2],dim=2)
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
            (targets_dx, targets_dy, targets_dw, targets_dh), dim=2
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