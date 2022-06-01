from .cls_head import ClassificationModel
from .reg_head import RegressionModel
from .retinanet_head import Retina_Head
from .test_head import test_Head
from .free_head import free_Head

def build_head(cfg, channel, num_anchor):

    head_name = cfg.Model.head.name.lower()
    if head_name == 'normal':
        # assert cfg.cls_head_out == cfg.Classes.num
        # build reg head
        regressionModel = RegressionModel(channel, num_anchor, cfg.reg_head)
        #build cls head
        classificationModel = ClassificationModel(channel, num_anchor, cfg.cls_head)

    elif head_name == 'retina_head':
        # head = eval(cfg.name)
        Head = Retina_Head(channel, num_anchor, cfg)
    elif head_name == 'test_head':
        # head = eval(cfg.name)
        Head = test_Head(channel, num_anchor, cfg)
    elif head_name == 'free_head':
        # head = eval(cfg.name)
        Head = free_Head(channel, cfg)
        
    else:
        raise NotImplementedError
    Head.init_weights()
    # classificationModel.init_weights()
    return Head
    
