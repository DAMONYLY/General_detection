'目前是用于构建模型的启动器，根据模型名字选择检测模型'

import torch
import torch.nn as nn
from model.anchor.retina_anchor import Retina_Anchors
from model.anchor.build_anchor import Anchors
from model.loss.build_loss import build_loss
from model.metrics.build_metrics import build_metrics
from model.loss.ultry_loss import FocalLoss

class Loss_calculater(nn.Module):
    def __init__(self, cfg) -> None:
        super(Loss_calculater, self).__init__()
        self.img_shape = cfg.TRAIN['TRAIN_IMG_SIZE']

        self.anchors = Anchors()
        self.retina_anchor = Retina_Anchors()
        # self.all_anchors = self.anchors(torch.zeros(size=(self.batch_size, 3, cfg.TRAIN['TRAIN_IMG_SIZE'],cfg.TRAIN['TRAIN_IMG_SIZE']),
        #                                 dtype=torch.double).cuda())
        self.label_assign = build_metrics(cfg, cfg.MODEL['metrics'])
        
        self.loss = build_loss(cfg.MODEL['loss'], cfg)
        
        self.FocalLoss = FocalLoss()
        
    def forward(self, imgs, features, targets=None):
        """
        Arguments:
            features (list[Tensor] or ImageList): features from head [reg, cls]
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the loss of model.
        """


        retina_anchors = self.retina_anchor(imgs).squeeze(0)
        proposals_reg, proposals_cls = features

        if False:
            losses, losses_reg, losses_cls = self.FocalLoss(proposals_cls, proposals_reg, retina_anchors.unsqueeze(0), targets)
        else:

            cls_pred, reg_pred, cls_target, reg_target = \
                                self.label_assign(retina_anchors, targets, proposals_reg, proposals_cls)

            losses, losses_reg, losses_cls = \
                                self.loss(cls_pred, reg_pred, cls_target, reg_target) # reg_loss, cls_loss, conf_loss

        return losses, losses_reg, losses_cls

    def flatten_features(self, anchors):
        """
        Args:
            anchors (list(torch.tensors)): the results of head output

        Returns: 
            anchors (list(torch.tensors)) : like [[B, N, w, h, feature_dim],...]
        """
        
        if len(anchors[0].shape) == 5:
            for id, item in enumerate(anchors):
                anchors[id] = item.contiguous().view(item.shape[0], -1, item.shape[-1])
            return torch.cat(anchors, dim = 1)
        elif len(anchors[0].shape) == 3:
            for id, item in enumerate(anchors):
                anchors[id] = item.view(-1, item.shape[-1])
            return torch.cat(anchors, dim = 0)