from cv2 import mean
from numpy.core.fromnumeric import shape
import torch.nn as nn
import torch
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)

        return loss

class Loss(nn.Module):
    def __init__(self, cls_loss, reg_loss):
        super(Loss, self).__init__()
        self.xy_loss = nn.SmoothL1Loss(reduction='sum')
        self.wh_loss = nn.SmoothL1Loss(reduction='sum')
        self.reg_loss = nn.SmoothL1Loss(reduction='mean')
        self.obj_loss = nn.BCELoss(reduction='mean')
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='mean')
        # self.test = nn.BCELoss()
    def forward(self, cls_pred, reg_pred, obj_pred, cls_target, reg_target, obj_target):
        obj_pred = torch.sigmoid(obj_pred)
        # loss_xy = self.xy_loss(reg_pred[..., :2], reg_target[..., :2])
        # loss_wh = self.wh_loss(reg_pred[..., 2:4], reg_target[..., 2:4])
        loss_reg = 10 * self.reg_loss(reg_pred, reg_target)
        # loss_obj =  (self.obj_loss(obj_target * obj_pred, obj_target) + \
                    # self.obj_loss((1 - obj_target) * obj_pred, (1 - obj_target) * obj_target))/obj_target.shape[0]
        loss_obj = self.obj_loss(obj_pred, obj_target) 
        loss_cls = self.cls_loss(cls_pred, cls_target)
        # loss_reg = loss_xy + loss_wh
        loss = loss_reg + loss_obj + loss_cls
        return loss, loss_reg, loss_obj, loss_cls
