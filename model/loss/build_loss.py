from model.loss.model_loss import *

def build_loss(cfg):
    if cfg.name == 'Reg_Cls_Loss':
        return Reg_Cls_Loss(cfg.cls_loss.name, cfg.reg_loss.name, cfg.cls_loss.ratio, cfg.reg_loss.ratio)
    
    elif cfg.name == 'Reg_Cls_Obj_Loss':
        assert 'obj_loss' in cfg
        return Reg_Cls_Obj_Loss(cfg.cls_loss.name, cfg.reg_loss.name, cfg.obj_loss.name,
                                cfg.cls_loss.ratio, cfg.reg_loss.ratio, cfg.obj_loss.ratio)
