#coding=utf-8
import sys
sys.path.append("..")
import torch
import numpy as np
import cv2
import random
import config.cfg_example as cfg
import os


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        print("initing {} ".format(m))
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('BatchNorm2d') != -1:
        print("initing {} ".format(m))

        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

    y[:, 0] = (x[:, 0] + x[:, 2]) / 2.0
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2.0
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


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

def iou_xyxy_torch_batch(boxes1, boxes2):
    """
    [B, N, 4] with [B, M, 4] return [B, N, M]
    """

    boxes1_area = torch.prod(boxes1[..., 2:] - boxes1[..., :2], 2)
    boxes2_area = torch.prod(boxes2[..., 2:] - boxes2[..., :2], 2)

    # 计算出boxes1与boxes2相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., None, :2], boxes2[:, None, :, :2])
    right_down = torch.min(boxes1[..., None, 2:], boxes2[:, None, :, 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(left_up))
    inter_area = torch.prod(inter_section, 3)

    union_area = boxes1_area[..., None] + boxes2_area[:, None, :] - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU



def IOU_xyxy_torch_same(boxes1, boxes2):
    """
    [N, 4] with [N, 4] return [N, 1]
    """

    boxes1_area = torch.prod(boxes1[:, 2:] - boxes1[:, :2], -1)
    boxes2_area = torch.prod(boxes2[:, 2:] - boxes2[:, :2], -1)

    # 计算出boxes1与boxes2相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[:, :2], boxes2[:, :2])
    right_down = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(left_up))
    inter_area = torch.prod(inter_section, -1)

    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

def GIOU_xyxy_torch_same(boxes1, boxes2):
    """
    [N, 4] with [N, 4] return [N, 1]
    """

    boxes1_area = torch.prod(boxes1[:, 2:] - boxes1[:, :2], -1)
    boxes2_area = torch.prod(boxes2[:, 2:] - boxes2[:, :2], -1)

    # 计算出boxes1与boxes2相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[:, :2], boxes2[:, :2])
    right_down = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(left_up))
    inter_area = torch.prod(inter_section, -1)

    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose_section = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_right_down))
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area
    return GIOU


def GIOU_xywh_torch(boxes1, boxes2):
    """
     https://arxiv.org/abs/1902.09630
    boxes1(boxes2)' shape is [..., (x,y,w,h)].The size is for original image.
    """
    # xywh->xyxy
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    inter_area =  inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose_section = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_right_down))
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area
    
    return GIOU
def Diou_xywh_torch(preds, bbox, eps=1e-7):
    '''
    preds:[[x,y,w,h],,,]
    bbox:[[x,y,w,h],,,]
    eps: eps to avoid divide 0
    reduction: mean or sum
    return: diou-loss
    '''
    # xywh->xyxy
    preds = torch.cat([preds[..., :2] - preds[..., 2:] * 0.5,
                        preds[..., :2] + preds[..., 2:] * 0.5], dim=-1)
    bbox = torch.cat([bbox[..., :2] - bbox[..., 2:] * 0.5,
                        bbox[..., :2] + bbox[..., 2:] * 0.5], dim=-1)

    preds = torch.cat([torch.min(preds[..., :2], preds[..., 2:]),
                        torch.max(preds[..., :2], preds[..., 2:])], dim=-1)
    bbox = torch.cat([torch.min(bbox[..., :2], bbox[..., 2:]),
                        torch.max(bbox[..., :2], bbox[..., 2:])], dim=-1)
    # calculate iou
    ix1 = torch.max(preds[..., 0], bbox[..., 0])
    iy1 = torch.max(preds[..., 1], bbox[..., 1])
    ix2 = torch.min(preds[..., 2], bbox[..., 2])
    iy2 = torch.min(preds[..., 3], bbox[..., 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
    inters = iw * ih

    # union
    uni = (preds[..., 2] - preds[..., 0] + 1.0) * (preds[..., 3] - preds[..., 1] + 1.0) + (bbox[..., 2] - bbox[..., 0] + 1.0) * (
            bbox[..., 3] - bbox[..., 1] + 1.0) - inters

    # iou
    iou = inters / (uni + eps)
    # print("iou:\n",iou)

    # inter_diag
    cxpreds = (preds[..., 2] + preds[..., 0]) / 2
    cypreds = (preds[..., 3] + preds[..., 1]) / 2

    cxbbox = (bbox[..., 2] + bbox[..., 0]) / 2
    cybbox = (bbox[..., 3] + bbox[..., 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
    # print("inter_diag:\n",inter_diag)

    # outer_diag
    ox1 = torch.min(preds[..., 0], bbox[..., 0])
    oy1 = torch.min(preds[..., 1], bbox[..., 1])
    ox2 = torch.max(preds[..., 2], bbox[..., 2])
    oy2 = torch.max(preds[..., 3], bbox[..., 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2
    # print("outer_diag:\n",outer_diag)

    diou = iou - inter_diag / outer_diag
    diou = torch.clamp(diou, min=-1.0, max=1.0)

    # diou_loss = 1 - diou
    # print("last_loss:\n",diou_loss)

    return diou

def CIOU_xywh_torch(preds, bbox, eps=1e-7):
    '''
    https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param eps: eps to avoid divide 0
    :param reduction: mean or sum
    :return: diou-loss
    '''
    import math
    # xywh to xyxy
    preds = torch.cat([preds[..., :2] - preds[..., 2:] * 0.5,
                        preds[..., :2] + preds[..., 2:] * 0.5], dim=-1)
    bbox = torch.cat([bbox[..., :2] - bbox[..., 2:] * 0.5,
                        bbox[..., :2] + bbox[..., 2:] * 0.5], dim=-1)

    preds = torch.cat([torch.min(preds[..., :2], preds[..., 2:]),
                        torch.max(preds[..., :2], preds[..., 2:])], dim=-1)
    bbox = torch.cat([torch.min(bbox[..., :2], bbox[..., 2:]),
                        torch.max(bbox[..., :2], bbox[..., 2:])], dim=-1)
    ix1 = torch.max(preds[..., 0], bbox[..., 0])
    iy1 = torch.max(preds[..., 1], bbox[..., 1])
    ix2 = torch.min(preds[..., 2], bbox[..., 2])
    iy2 = torch.min(preds[..., 3], bbox[..., 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
    inters = iw * ih

    # union
    uni = (preds[..., 2] - preds[..., 0] + 1.0) * (preds[..., 3] - preds[..., 1] + 1.0) + (bbox[..., 2] - bbox[..., 0] + 1.0) * (
            bbox[..., 3] - bbox[..., 1] + 1.0) - inters

    # iou
    iou = inters / (uni + eps)
    # print("iou:\n",iou)

    # inter_diag
    cxpreds = (preds[..., 2] + preds[..., 0]) / 2
    cypreds = (preds[..., 3] + preds[..., 1]) / 2

    cxbbox = (bbox[..., 2] + bbox[..., 0]) / 2
    cybbox = (bbox[..., 3] + bbox[..., 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

    # outer_diag
    ox1 = torch.min(preds[..., 0], bbox[..., 0])
    oy1 = torch.min(preds[..., 1], bbox[..., 1])
    ox2 = torch.max(preds[..., 2], bbox[..., 2])
    oy2 = torch.max(preds[..., 3], bbox[..., 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

    diou = iou - inter_diag / outer_diag
    # print("diou:\n",diou)

    # calculate v,alpha
    wbbox = bbox[..., 2] - bbox[..., 0] + 1.0
    hbbox = bbox[..., 3] - bbox[..., 1] + 1.0
    wpreds = preds[..., 2] - preds[..., 0] + 1.0
    hpreds = preds[..., 3] - preds[..., 1] + 1.0
    v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
    alpha = v / (1 - iou + v)
    ciou = diou - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    # ciou_loss = 1 - ciou
    # if reduction == 'mean':
    #     loss = torch.mean(ciou_loss)
    # elif reduction == 'sum':
    #     loss = torch.sum(ciou_loss)
    # else:
    #     raise NotImplementedError
    # print("last_loss:\n",loss)
    return ciou

def nms(bboxes, score_threshold, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes:
    假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    :return: best_bboxes
    假设NMS后剩下N个bbox，那么best_bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    """
    classes_in_img = list(set(bboxes[:, 5].astype(np.int32)))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5].astype(np.int32) == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = iou_xyxy_numpy(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            assert method in ['nms', 'soft-nms']
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    return np.array(best_bboxes)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_box(bboxes, img, id = None, color=None, line_thickness=None):
    """
    显示图片img和其所有的bboxes
    :param bboxes: [N, 5] 表示N个bbox, 格式仅支持np.array
    :param img: img格式为pytorch, 需要进行转换
    :param color:
    :param line_thickness:
    """

    img = img.permute(0,2,3,1).contiguous()[0].numpy() if isinstance(img, torch.Tensor) else img# [C,H,W] ---> [H,W,C]
    img_size, _, _ = img.shape
    bboxes[:, :4] = xywh2xyxy(bboxes[:, :4])
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    for i, x in enumerate(bboxes):
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        label = cfg.DATA["CLASSES"][int(x[4])]
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    # cv2.imshow("img-bbox", img[:, :, ::-1])
    # cv2.waitKey(0)
    img = cv2.cvtColor(img* 255.0, cv2.COLOR_RGB2BGR).astype(np.float32)
    cv2.imwrite("../data/dataset{}.jpg".format(id), img)
