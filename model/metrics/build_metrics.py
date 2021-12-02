'''
2021年11月29日16:58:22
用于 label assign 包括centerness, iou, ATSS

'''
from model.metrics.general_metrics import label_assign


def build_metrics(type):
    if type == 'yolo':
        # metrics = label_assign('iou')
        return label_assign('iou')
    else:
        raise NotImplementedError