'''
2021年11月29日17:00:15
通用label assign
yolo形式
'''
import torch.nn as nn
import torch
import sys
from model.anchor.build_anchor import Anchors
sys.path.append('/raid/yaoliangyong/General_detection/')
from utils.tools import *



class label_assign(nn.Module):
    def __init__(self, cfg, metrics_type, pos_iou_thr = 0.5, neg_iou_thr = 0.3) -> None:
        super(label_assign, self).__init__()
        self.metrics = metrics_type
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.strides = [32, 16, 8]
        # self.anchor = self.origin_to_grid(cfg.MODEL['ANCHORS'], cfg.TRAIN['TRAIN_IMG_SIZE'], self.strides)
        # self.anchor = torch.tensor(cfg.MODEL['ANCHORS'])
        # self.num_anchor = self.anchor.shape[1]


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
        obj_preds_assign = []
        cls_targets_assign = []
        reg_targets_assign = []
        obj_targets_assign = []

        batch_size = targets.shape[0]

        device = targets.device
        dtype = targets.dtype

        anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1)
        
        for batch_id in range(batch_size):
            classification = classifications[batch_id]
            regression, objecteness = regressions[batch_id][..., :4], regressions[batch_id][..., -1:]
            anchor = anchors[batch_id].type(dtype).to(device)
            target = targets[batch_id]
            target = target[target[:, 4] != -1]

            num_anchor = anchor.shape[0]

            # 计算IOU between anchor and target
            # overlaps = iou_xyxy_torch_batch(anchor, bbox_annotation)
            overlaps = iou_xyxy_torch(anchor, target[..., :4]) # [N, M]
            
            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            neg_mask = max_overlaps < self.neg_iou_thr
            pos_mask = max_overlaps > self.pos_iou_thr
            

            assign_targets = target[argmax_overlaps]
            pos_assign_targets = assign_targets[pos_mask]
            
            # bulid cls targets and preds
            # classification = classification.contiguous().view(batch_size, num_anchor, classification.shape[-1])
            cls_preds = classification[pos_mask]

            cls_targets = torch.zeros_like(cls_preds)
            
            cls_targets[torch.arange(cls_targets.shape[0]), pos_assign_targets[..., 4].long()] = 1

            # bulid reg targets and preds
            reg_preds = regression[pos_mask]

            reg_targets = self.encode(anchor[pos_mask], pos_assign_targets)

            # bulid obj targets and preds

            pos_obj_preds = objecteness[pos_mask]
            neg_obj_preds = objecteness[neg_mask]

            pos_obj_targets = torch.ones_like(pos_obj_preds)
            neg_obj_targets = torch.zeros_like(neg_obj_preds)

            obj_preds = torch.cat((pos_obj_preds, neg_obj_preds), dim = 0)
            obj_targets =torch.cat((pos_obj_targets, neg_obj_targets), dim = 0)
            
            cls_preds_assign.append(cls_preds)
            reg_preds_assign.append(reg_preds)
            obj_preds_assign.append(obj_preds)
            cls_targets_assign.append(cls_targets)
            reg_targets_assign.append(reg_targets)
            obj_targets_assign.append(obj_targets)

        cls_preds_assign = torch.cat(cls_preds_assign)
        reg_preds_assign = torch.cat(reg_preds_assign)
        obj_preds_assign = torch.cat(obj_preds_assign)
        cls_targets_assign = torch.cat(cls_targets_assign)
        reg_targets_assign = torch.cat(reg_targets_assign)
        obj_targets_assign = torch.cat(obj_targets_assign)

        return cls_preds_assign, reg_preds_assign, obj_preds_assign, cls_targets_assign, reg_targets_assign, obj_targets_assign

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

    def origin_to_grid(self, anchor, img_shape, strides):
        """
        将anchor转化成grid形式的anchor，用于计算iou
        """
        base_anchor = []
        anchors = torch.tensor(anchor)
        for id, stride in enumerate(strides):
            anchor = anchors[id]
            widths = anchor[..., 0]
            heights = anchor[..., 1]
            feature_w = int(img_shape / stride)
            feature_h = int(img_shape / stride)
            x = (torch.arange(0, feature_w).unsqueeze(1) + 0.5).repeat(1, feature_h)
            y = (torch.arange(0, feature_h).unsqueeze(0) + 0.5).repeat(feature_w, 1)
            grid_xy = torch.stack([x, y], dim=-1)
            anchor =  torch.Tensor([grid_xy[..., 0] - 0.5 * widths, 
                                         grid_xy[..., 1] - 0.5 * heights, 
                                         grid_xy[..., 0] + 0.5 * widths, 
                                         grid_xy[..., 1] + 0.5 * heights])
            base_anchor.append(anchor)
        return torch.stack(base_anchor)

    def anchor_to_target(self, anchor, target_grids):
        """
        anchor [3,2]
        target_graids [M, 2]
        """
        num_anchor = anchor.shape[0]
        num_target = target_grids.shape[0]
        target_x = (target_grids[..., 0] + 0.5).repeat(num_anchor, 1)
        target_y = (target_grids[..., 1] + 0.5).repeat(num_anchor, 1)
        anchor_w = anchor[..., 0].unsqueeze(1).repeat(1, num_target)
        anchor_h = anchor[..., 1].unsqueeze(1).repeat(1, num_target)
        base_anchor = torch.stack([target_x - 0.5 * anchor_w, 
                                    target_y - 0.5 * anchor_h, 
                                    target_x + 0.5 * anchor_w, 
                                    target_y + 0.5 * anchor_h], dim = -1)
        return base_anchor

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

    def encode(self, anchor, gt_bbox, eps = 1e-6):
        
        # dtype = gt_bbox.dtype
        # change [xmin, ymin, xmax, ymax] to [x_center, y_center, width, height]
        anchor_widths = anchor[..., 2] - anchor[..., 0]
        anchor_heights = anchor[..., 3] - anchor[..., 1]
        # ex_widths = torch.clamp(ex_widths, min=1)
        # ex_heights = torch.clamp(ex_heights, min=1)
        anchor_ctr_x = anchor[..., 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[..., 1] + 0.5 * anchor_heights
        # ex_widths = ex_rois[..., 0]
        # ex_heights = ex_rois[..., 1]

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