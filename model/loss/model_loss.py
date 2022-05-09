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
        # focal_weight = torch.where(torch.eq(target, 1.), 1. - input, input)
        focal_weight = torch.where(torch.ge(target, 0.5), 1. - input, input)
        loss *= self.__alpha * torch.pow(focal_weight, self.__gamma)

        # return loss.sum()/torch.clamp(torch.sum(target == 1).float(), min=1.0)
        return loss.sum()/torch.clamp(torch.sum(target >= 0.5).float(), min=1.0)
class IOU_Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super(IOU_Loss, self).__init__()
        self.reduction = reduction
    def forward(self, input, target):
        loss = target - input
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

class Loss(nn.Module):
    def __init__(self, cls_loss, reg_loss, cls_ratio=1, reg_ratio=1):
        super(Loss, self).__init__()
        if cls_loss == 'Focal':
            self.cls_loss = Focal_Loss()
        elif cls_loss == 'BCEWithLogitsLoss':
            self.cls_loss = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            raise NotImplementedError
        self.cls_ratio = cls_ratio

        if reg_loss == 'SmoothL1':
            self.reg_loss = nn.SmoothL1Loss(reduction='mean')
        elif reg_loss == 'IOU':
            self.reg_loss = IOU_Loss()
        else:
            raise NotImplementedError
        self.reg_ratio = reg_ratio

    def forward(self, cls_pred, reg_pred, cls_target, reg_target):

        loss_reg = self.reg_ratio * self.reg_loss(reg_pred, reg_target)
        loss_cls = self.cls_ratio * self.cls_loss(cls_pred, cls_target)

        loss = loss_reg + loss_cls
        return loss, loss_reg, loss_cls
