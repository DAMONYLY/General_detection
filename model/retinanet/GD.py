'''
2021年11月29日17:00:15
通用label assign
yolo形式
'''
from torchvision.ops import nms
import torch.nn as nn
import torch
import sys
sys.path.append('/raid/yaoliangyong/General_detection/')

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

'''
2021年11月29日17:00:15
通用label assign
yolo形式
'''
import torch.nn as nn
import torch
import sys




class label_assign(nn.Module):
    def __init__(self, cfg, metrics_type='iou', pos_iou_thr = 0.5, neg_iou_thr = 0.3) -> None:
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


            # 计算IOU between anchor and target
            # overlaps = iou_xyxy_torch_batch(anchor, bbox_annotation)
            overlaps = iou_xyxy_torch(anchor, target[..., :4]) # [N, M]
            
            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            neg_mask = max_overlaps < self.neg_iou_thr
            pos_mask = max_overlaps > self.pos_iou_thr
            

            assign_targets = target[argmax_overlaps]

            
            # bulid cls targets and preds
            # 
            pos_cls_preds = classification[pos_mask]
            pos_cls_targets = torch.zeros_like(pos_cls_preds)
            
            neg_cls_preds = classification[neg_mask]
            neg_cls_targets = torch.zeros_like(neg_cls_preds)
            pos_cls_targets[torch.arange(pos_cls_targets.shape[0]), assign_targets[pos_mask, 4].long()] = 1

            cls_preds = torch.cat([pos_cls_preds, neg_cls_preds], dim = 0)
            cls_targets = torch.cat([pos_cls_targets, neg_cls_targets], dim = 0)
            # bulid reg targets and preds
            reg_preds = regression[pos_mask]
            reg_targets = self.encode(anchor[pos_mask], assign_targets[pos_mask])


            
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
class Focal_Loss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="none"):
        super(Focal_Loss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        # self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.__loss = nn.BCELoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        focal_weight = torch.where(torch.eq(target, 1.), 1. - input, input)
        loss *= self.__alpha * torch.pow(focal_weight, self.__gamma)

        return loss.sum()/torch.sum(target == 1)


class Loss(nn.Module):
    def __init__(self, cls_loss, reg_loss):
        super(Loss, self).__init__()
        self.xy_loss = nn.SmoothL1Loss(reduction='sum')
        self.wh_loss = nn.SmoothL1Loss(reduction='sum')
        self.reg_loss = nn.SmoothL1Loss(reduction='mean')
        self.obj_loss = nn.BCELoss(reduction='mean')
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.cls_loss_FC = Focal_Loss()
        # self.test = nn.BCELoss()
    def forward(self, cls_pred, reg_pred, cls_target, reg_target):

        loss_reg = 10 * self.reg_loss(reg_pred, reg_target)
        # loss_reg = self.reg_loss_FC(reg_pred, reg_target)
        # loss_cls = self.cls_loss(cls_pred, cls_target)
        loss_cls = self.cls_loss_FC(cls_pred, cls_target)

        loss = loss_reg  + loss_cls
        return loss, loss_reg, loss_cls

def yolo_decode(feature, anchor):
    """
    Args:
        feature (torch.tensor): output of fpn. [B, num_per_anchor, W, H, reg].
        anchor (torch.tensor): anchors [num_per_grid, feature_w * feature_h, 4]

    output:
        feature (torch.tensor): after decode. [all_num_anchors, reg], in xyxy form.
    """
    batch_size, num_anchor, out_dim = feature.shape
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
    # pred_obj = torch.sigmoid(feature[..., 4:5])
    # pred_cls = torch.sigmoid(feature[..., 5:])

    # pred_bbox = torch.cat([pred_reg, pred_obj, pred_cls], dim=-1).view(-1, out_dim)

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