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
        # build reg head
        regressionModel = RegressionModel(channel, num_anchor, cfg.reg_head)
        
        #build cls head
        classificationModel = ClassificationModel(channel, num_anchor, cfg.cls_head)

    else:
        raise NotImplementedError
    regressionModel.init_weights()
    classificationModel.init_weights()
    return regressionModel, classificationModel
    
