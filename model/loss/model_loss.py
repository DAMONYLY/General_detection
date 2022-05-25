import torch.nn as nn
import torch
import torch.nn.functional as F
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

    def forward(self, input, target, weight, num_pos):

        loss = self.loss(input=input, target=target)
        focal_weight = target * input + (1 - target) * (1 - input)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss *= alpha * torch.pow(1.0 - focal_weight, self.gamma)
        
        loss *= weight
        assert self.reduction in ['none', 'mean', 'sum', 'avg_pos']
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'avg_pos':
            test = torch.clamp(torch.sum(target >= 0.5).float(), min=1.0)
            return loss.sum()/num_pos

class L1_Loss(nn.Module):
    def __init__(self, reduction="avg_pos"):
        super(L1_Loss, self).__init__()
        self.reduction = reduction
        self.loss = nn.SmoothL1Loss(reduction='none')
    def forward(self, input, target, weight, num_pos):
        loss = self.loss(input, target)
        
        loss *= weight
        assert self.reduction in ['none', 'mean', 'sum', 'avg_pos']
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'avg_pos':
            return loss.sum()/num_pos
 
class IOU_Loss(nn.Module):
    def __init__(self, reduction="avg_pos"):
        super(IOU_Loss, self).__init__()
        self.reduction = reduction
    def forward(self, input, target, weight, num_pos, mode='linear'):
        if mode == 'linear':
            loss = 1 - target
        elif mode == 'square':
            loss = 1 - target**2
        elif mode == 'log':
            loss = -target.log()
        else:
            raise NotImplementedError
        
        loss *= weight
        assert self.reduction in ['none', 'mean', 'sum', 'avg_pos']
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'avg_pos':
            return loss.sum()/num_pos

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
            self.reg_loss = L1_Loss()
        elif reg_loss == 'IOU':
            self.reg_loss = IOU_Loss()
        else:
            raise NotImplementedError
        self.reg_ratio = reg_ratio

    def forward(self, reg_pred, reg_targets, reg_weights, cls_pred, cls_targets, cls_weights, num_pos_inds):
        
        loss_reg = self.reg_ratio * self.reg_loss(reg_pred, reg_targets, reg_weights, num_pos_inds)
        loss_cls = self.cls_ratio * self.cls_loss(cls_pred, cls_targets, cls_weights, num_pos_inds)

        loss = loss_reg + loss_cls
        return loss, loss_reg, loss_cls
