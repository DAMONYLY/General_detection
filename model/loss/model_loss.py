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

class Loss():
    def __init__(self, cls_loss, reg_loss):
        super(Loss, self).__init__()
        self.x_y_loss = nn.BCELoss()
        self.w_h_loss = nn.MSELoss()
        self.cls_loss = nn.BCELoss()

    def forward(self, indices, cls_pred, cls_targets, reg_pred, reg_targets):
        print('1')
        return 1
