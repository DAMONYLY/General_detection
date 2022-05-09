'''
2021年11月29日16:58:22
用于 label assign 包括centerness, iou, ATSS

'''
from ..metrics import Max_iou_assigner


def build_metrics(cfg):
    if cfg.Model.metrics.name == 'Max_iou':
        pos_iou_thr = getattr(cfg.Model.metrics, 'pos_iou_thr', 0.5)
        neg_iou_thr = getattr(cfg.Model.metrics, 'neg_iou_thr', 0.3)
        return Max_iou_assigner(cfg, pos_iou_thr, neg_iou_thr)
    else:
        raise NotImplementedError