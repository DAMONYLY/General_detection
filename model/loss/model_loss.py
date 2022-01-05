import torch.nn as nn
import torch
class Focal_Loss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="none"):
        super(Focal_Loss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        # self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.__loss = nn.BCELoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        focal_weight = torch.where(torch.eq(target, 1.), 1. - input, input)
        loss *= self.__alpha * torch.pow(focal_weight, self.__gamma)

        return loss.sum()/torch.clamp(torch.sum(target == 1).float(), min=1.0)


class Loss(nn.Module):
    def __init__(self, cls_loss, reg_loss):
        super(Loss, self).__init__()
        self.xy_loss = nn.SmoothL1Loss(reduction='sum')
        self.wh_loss = nn.SmoothL1Loss(reduction='sum')
        self.reg_loss = nn.SmoothL1Loss(reduction='mean')
        self.obj_loss = nn.BCELoss(reduction='mean')
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.cls_loss_FC = Focal_Loss()
        # self.test = nn.BCELoss()
    def forward(self, cls_pred, reg_pred, cls_target, reg_target):

        loss_reg = 10 * self.reg_loss(reg_pred, reg_target)
        # loss_reg = self.reg_loss_FC(reg_pred, reg_target)
        # loss_cls = self.cls_loss(cls_pred, cls_target)
        loss_cls = self.cls_loss_FC(cls_pred, cls_target)

        loss = loss_reg  + loss_cls
        return loss, loss_reg, loss_cls
