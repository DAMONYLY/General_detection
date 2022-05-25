from .cls_head import ClassificationModel
from .reg_head import RegressionModel
from .retinanet_head import Retinanethead

def build_head(cfg, channel, num_anchor):

    if cfg.name.lower() == 'normal':
        # assert cfg.cls_head_out == cfg.Classes.num
        # build reg head
        regressionModel = RegressionModel(channel, num_anchor, cfg.reg_head)
        #build cls head
        classificationModel = ClassificationModel(channel, num_anchor, cfg.cls_head)

    elif cfg.name.lower() == 'retinanethead':
        # head = eval(cfg.name)
        head = Retinanethead(channel, num_anchor, cfg)
        # classificationModel = ClassificationModel(channel, num_anchor, cfg.cls_head)
        
    else:
        raise NotImplementedError
    head.init_weights()
    # classificationModel.init_weights()
    return head
    
