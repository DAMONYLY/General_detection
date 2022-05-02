'''
2021年11月29日17:16:02
用于建立loss函数
输入，pred box ，label box
输出：回归loss， 分类loss， 置信度loss
'''
from model.loss.model_loss import Loss


def build_loss(cfg):
    return Loss(cfg.cls_loss.name, cfg.reg_loss.name, cfg.cls_loss.ratio, cfg.reg_loss.ratio)
    # if type == 'fun':
    #     return Loss(cfg, 1)
    # else:
    #     raise NotImplementedError