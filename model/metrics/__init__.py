from .Max_iou_metrics import Max_iou_assigner
from .ATSS_metrics import ATSSAssigner

def build_metrics(cfg):
    if cfg.Model.metrics.name.lower() == 'max_iou':
        pos_iou_thr = getattr(cfg.Model.metrics, 'pos_iou_thr', 0.5)
        neg_iou_thr = getattr(cfg.Model.metrics, 'neg_iou_thr', 0.3)
        return Max_iou_assigner(cfg, pos_iou_thr, neg_iou_thr)
    elif cfg.Model.metrics.name.lower() == 'atss':
        topk = getattr(cfg.Model.metrics, 'topk', 9)
        return ATSSAssigner(topk)

    else:
        raise NotImplementedError