import torch.nn as nn
import torch
class Focal_Loss(nn.Module):
    """Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Args:
        
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction="avg_pos"):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, input, target):
        loss = self.loss(input=input, target=target)
        # focal_weight = torch.where(torch.ge(target, 0.5), input, 1. - input)
        focal_weight = target * input + (1 - target) * (1 - input)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss *= alpha * torch.pow(1.0 - focal_weight, self.gamma)
        assert self.reduction in ['none', 'mean', 'sum', 'avg_pos']
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'avg_pos':
            num_pos = torch.clamp(torch.sum(target >= 0.5).float(), min=1.0)
            return loss.sum()/num_pos
    
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
        elif cls_loss == 'BCELoss':
            self.cls_loss = nn.BCELoss(reduction='mean')
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
