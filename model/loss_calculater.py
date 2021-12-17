'目前是用于构建模型的启动器，根据模型名字选择检测模型'
"2021年11月24日20:31:54"

import torch.nn as nn
from model.anchor.build_anchor import Anchors
from model.loss.build_loss import build_loss
from model.metrics.build_metrics import build_metrics

class Loss_calculater(nn.Module):
    def __init__(self, cfg) -> None:
        super(Loss_calculater, self).__init__()
        
        self.anchors = Anchors()
        # self.all_anchors = self.anchors(torch.zeros(size=(self.batch_size, 3, cfg.TRAIN['TRAIN_IMG_SIZE'],cfg.TRAIN['TRAIN_IMG_SIZE']),
        #                                 dtype=torch.double).cuda())
        self.label_assign = build_metrics(cfg, cfg.MODEL['metrics'])
        
        self.loss = build_loss(cfg.MODEL['loss'], cfg)
        
    def forward(self, features, targets=None):
        """
        Arguments:
            features (list[Tensor] or ImageList): features from fpn. [large, medium, small]
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the loss of model.
        """

        # anchors = self.anchors(image = images, only_anchors = True)
        anchors = []
        proposals_reg, proposals_cls = features
        cls_pred, reg_pred, obj_pred, cls_target, reg_target, obj_target = \
                            self.label_assign(anchors, targets, proposals_reg, proposals_cls)

        losses, losses_reg, losses_obj, losses_cls = \
                            self.loss(cls_pred, reg_pred, obj_pred, cls_target, reg_target, obj_target) # reg_loss, cls_loss, conf_loss

        return losses, losses_reg, losses_obj, losses_cls
