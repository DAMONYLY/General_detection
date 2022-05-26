import torch.nn as nn
from .loss import *

class Reg_Cls_Loss(nn.Module):
    def __init__(self, cls_loss, reg_loss, cls_ratio=1, reg_ratio=1):
        super(Reg_Cls_Loss, self).__init__()

        assert cls_loss in ['Focal_Loss', 'BCE_Loss'], f'Unsupport cls loss type'
        self.cls_loss = eval(cls_loss)()
        self.cls_ratio = cls_ratio

        assert reg_loss in ['SmoothL1_Loss', 'IOU_Loss'], f'Unsupport reg loss type'
        self.reg_loss = eval(reg_loss)()
        self.reg_ratio = reg_ratio

    def forward(self, reg_pred, reg_targets, reg_weights, cls_pred, cls_targets, cls_weights, num_pos_inds):
        
        loss_reg = self.reg_ratio * self.reg_loss(reg_pred, reg_targets, reg_weights, num_pos_inds)
        loss_cls = self.cls_ratio * self.cls_loss(cls_pred, cls_targets, cls_weights, num_pos_inds)

        loss = loss_reg + loss_cls
        return loss, loss_reg, loss_cls
    
class Reg_Cls_Obj_Loss(nn.Module):
    def __init__(self, cls_loss, reg_loss, obj_loss, cls_ratio=1, reg_ratio=1, obj_ratio=1):
        super(Reg_Cls_Obj_Loss, self).__init__()

        assert reg_loss in ['SmoothL1_Loss', 'IOU_Loss'], f'Unsupport reg loss type'
        self.reg_loss = eval(reg_loss)()
        self.reg_ratio = reg_ratio
        
        assert cls_loss in ['Focal_Loss', 'BCE_Loss'], f'Unsupport cls loss type'
        self.cls_loss = eval(cls_loss)()
        self.cls_ratio = cls_ratio

        assert obj_loss in ['BCE_Loss'], f'Unsupport obj loss type'
        self.obj_loss = eval(obj_loss)()
        self.obj_ratio = obj_ratio

    def forward(self, reg_pred, reg_targets, reg_weights, 
                      cls_pred, cls_targets, cls_weights, 
                      obj_pred, obj_targets, obj_weights,
                      num_pos_inds):
        
        loss_reg = self.reg_ratio * self.reg_loss(reg_pred, reg_targets, reg_weights, num_pos_inds)
        loss_cls = self.cls_ratio * self.cls_loss(cls_pred, cls_targets, cls_weights, num_pos_inds)
        loss_obj = self.obj_ratio * self.obj_loss(obj_pred, obj_targets, obj_weights, num_pos_inds)

        loss = loss_reg + loss_cls + loss_obj
        return loss, loss_reg, loss_cls, loss_obj
