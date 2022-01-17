'''
2021年11月29日16:19:14
head 首先分共享不共享两种形式
共享，则直接建一个重复使用即可
不共享，则需要多建几次，再使用

其次，按构成，分为回归头，分类头，置信度头
无置信度头，则放在分类头一起
'''
from .cls_head import ClassificationModel
from .reg_head import RegressionModel

def build_head(cfg, channel, num_anchor):

    if cfg.name == 'normal':
        # assert cfg.cls_head_out == cfg.Classes.num
        reg_channel = cfg.reg_head_out
        cls_channel = cfg.cls_head_out
        regressionModel = RegressionModel(channel, num_anchor)
        classificationModel = ClassificationModel(channel, num_anchor, num_classes=cls_channel)
        return regressionModel, classificationModel
    else:
        raise NotImplementedError
    
