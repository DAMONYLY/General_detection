from cv2 import mean
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
        self.x_y_loss = nn.BCELoss()
        self.w_h_loss = nn.MSELoss()
        self.cls_loss = nn.BCEWithLogitsLoss()

    def forward(self, cls_pred, reg_pred, cls_target, reg_target):

        loss_xy = self.x_y_loss(torch.sigmoid(reg_pred[..., :2]), reg_target[..., :2])
        loss_wh = self.w_h_loss(reg_pred[..., 2:4], reg_target[..., 2:4])
        loss_cls = self.cls_loss(cls_pred, cls_target)
        loss = loss_xy + loss_wh + loss_cls
        return loss, loss_xy, loss_wh, loss_cls
