from .cls_head import ClassificationModel
from .reg_head import RegressionModel
from .retinanet_head import Retina_Head

def build_head(cfg, channel, num_anchor):

    if cfg.name.lower() == 'normal':
        # assert cfg.cls_head_out == cfg.Classes.num
        # build reg head
        regressionModel = RegressionModel(channel, num_anchor, cfg.reg_head)
        #build cls head
        classificationModel = ClassificationModel(channel, num_anchor, cfg.cls_head)

    elif cfg.name.lower() == 'retina_head':
        # head = eval(cfg.name)
        Head = Retina_Head(channel, num_anchor, cfg)
        
    else:
        raise NotImplementedError
    Head.init_weights()
    # classificationModel.init_weights()
    return Head
    
