'目前是用于构建模型的启动器，根据模型名字选择检测模型'

from numpy import dtype
import torch
import torch.nn as nn
from model.anchor.build_anchor import Anchors
from model.loss.build_loss import build_loss
from model.metrics.build_metrics import build_metrics


class Loss_calculater(nn.Module):
    """
    Post-processing of the features extracted by the detector.
    Includes label assign and loss calculation.
    """
    def __init__(self, cfg) -> None:
        super(Loss_calculater, self).__init__()
        self.anchors = Anchors(cfg.Model.anchors)
        self.label_assign = build_metrics(cfg, cfg.Model.metrics)
        self.loss = build_loss(cfg.Model.loss)
        self.img_size = cfg.Data.train.pipeline.input_size
        
    def forward(self, imgs, features, targets=None):
        """
        Arguments:
            features (list[Tensor] or ImageList): features from head [reg, cls]
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the loss of model.
        """

        proposals_reg, proposals_cls = features
        
        assert self.img_size[0] == imgs.size()[-2:][0] and \
               self.img_size[1] == imgs.size()[-2:][1]
        anchors = self.anchors(self.img_size, device=proposals_reg.device, dtype=proposals_reg.dtype)

        cls_pred, reg_pred, cls_target, reg_target = \
                            self.label_assign(anchors, targets, proposals_reg, proposals_cls)

        losses, losses_reg, losses_cls = \
                            self.loss(cls_pred, reg_pred, cls_target, reg_target) # reg_loss, cls_loss, conf_loss

        return losses, losses_reg, losses_cls

