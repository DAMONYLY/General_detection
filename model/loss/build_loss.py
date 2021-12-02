'''
2021年11月29日17:16:02
用于建立loss函数
输入，pred box ，label box
输出：回归loss， 分类loss， 置信度loss
'''
from model.loss.model_loss import Loss
from model.loss.yolo_loss import YoloV3Loss


def build_loss(type, cfg):
    if type == 'test':
        return YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
    elif type == 'fun':
        return Loss(1, 1)
    else:
        raise NotImplementedError