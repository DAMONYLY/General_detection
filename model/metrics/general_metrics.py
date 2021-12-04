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
    def __init__(self, metrics_type, pos_iou_thr = 0.3, neg_iou_thr = 0.3) -> None:
        super(label_assign, self).__init__()
        self.metrics = metrics_type
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr



    def forward(self, anchor, target, regressions, classifications):
        # TODO
        """
        Arguments:
            anchors : [N,4] xyxy
            target: [B, M, 6] 6 contains xyxy,ind,mixind
            classifications: [B,N,20] model cls pred, 20指类别
            regressions: [B, N, 5] model reg pred
        
            一个anchor 最多一个anchor， 一个GT至少有一个anchor
        """
        cls_label_assign = []
        reg_label_assign = []
        positive_indices_all = []
        # anchor = anchors[:, :]
        batch_size = target.shape[0]
        device = target.device
        # classifications = torch.rand((1, 150, 20))

        for idx in range(batch_size):

            classification = classifications[idx, :, :]
            regression = regressions[idx, :, :]
            bbox_annotation = target[idx]
            
            # bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                # 表示为空，没有目标
                continue
            # 计算IOU between anchor and target
            overlaps = iou_xyxy_torch(anchor, bbox_annotation[:, :4])
            
            # overlaps = calc_hbb_iou(anchor[:, :], bbox_annotation[:, :4], mode='iou')
            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            # -------------- compute the classification loss ------------------------------ #
            cls_targets = torch.zeros_like(classification).to(device=device)
            # 比下界的置信度小，则为负样本，置0
            # cls_targets[torch.lt(max_overlaps, self.neg_iou_thr), :] = 0
            # 最大iou是否比正样本的iou界大
            positive_indices = torch.ge(max_overlaps, self.pos_iou_thr)

            num_positive_anchors = positive_indices.sum()
            #每一行包括当前行的anchor所匹配上的GT box
            assigned_annotations = bbox_annotation[argmax_overlaps, :]
            # cls_targets[positive_indices, :] = 0
            # 对应类别赋值为1
            cls_targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            # ---------------------------- compute regression loss -------------------------------- #
            reg_targets = torch.zeros_like(regression).to(device=device)
            # print('num_positive_anchors:',num_positive_anchors)
            if num_positive_anchors > 0:
                positive_anchors = anchor[positive_indices, :]
                gt_boxes = assigned_annotations[positive_indices, :]
                reg_target = self.encode(positive_anchors, gt_boxes)
            else:
                reg_target = torch.tensor(0).double().cuda(device=device)
            reg_targets[positive_indices, :4] = reg_target
            cls_label_assign.append(cls_targets)
            reg_label_assign.append(reg_targets)
            positive_indices_all.append(positive_indices)
        # calculate mean cls loss & mean reg loss of per batch size
        cls_label_assign = torch.stack(cls_label_assign)
        reg_label_assign = torch.stack(reg_label_assign)
        positive_indices_all = torch.stack(positive_indices_all)

        return positive_indices_all, cls_label_assign, reg_label_assign


    def encode(self, ex_rois, gt_rois):
        dtype = gt_rois.dtype
        # change [xmin, ymin, xmax, ymax] to [x_center, y_center, width, height]
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
        ex_widths = torch.clamp(ex_widths, min=1)
        ex_heights = torch.clamp(ex_heights, min=1)
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
        gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
        gt_widths = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)
        gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

        # targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        # targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dx = gt_ctr_x - ex_ctr_x.to(torch.int16).to(dtype)
        targets_dy = gt_ctr_y - ex_ctr_y.to(torch.int16).to(dtype)

        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)

        targets = torch.stack(
            (targets_dx, targets_dy, targets_dw, targets_dh), dim=1
        )
        return targets
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