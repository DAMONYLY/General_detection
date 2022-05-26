import torch
import torch.nn as nn

def weight_reduce_loss(loss, weight=None, reduction='avg_by_pos', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'avg_by_pos' and avg_factor is not None:
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)
    else:
        raise ValueError('Unsupport type of loss reduce')
    return loss


class Focal_Loss(nn.Module):
    """Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Args:
        
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction="avg_by_pos"):
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
        
        loss = weight_reduce_loss(loss=loss, weight=weight, reduction=self.reduction, avg_factor=num_pos)
        return loss

class BCE_Loss(nn.Module):
    """
        
    """
    def __init__(self, reduction="avg_by_pos"):
        super(BCE_Loss, self).__init__()
        self.reduction = reduction
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, input, target, weight, num_pos):

        loss = self.loss(input=input, target=target)
        
        loss = weight_reduce_loss(loss=loss, weight=weight, reduction=self.reduction, avg_factor=num_pos)
        return loss

class SmoothL1_Loss(nn.Module):
    """SmoothL1 loss for reg

    Args:
        
    """
    def __init__(self, reduction="avg_by_pos"):
        super(SmoothL1_Loss, self).__init__()
        self.reduction = reduction
        self.loss = nn.SmoothL1Loss(reduction='none')
    def forward(self, input, target, weight, num_pos):
        loss = self.loss(input, target)
        loss = weight_reduce_loss(loss=loss, weight=weight, reduction=self.reduction, avg_factor=num_pos)
        return loss
        
class IOU_Loss(nn.Module):
    """IOU loss for reg

    Args:
        
    """
    def __init__(self, reduction="avg_by_pos"):
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
        
        loss = weight_reduce_loss(loss=loss, weight=weight, reduction=self.reduction, avg_factor=num_pos)
        return loss