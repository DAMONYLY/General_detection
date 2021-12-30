'目前是用于构建模型的启动器，根据模型名字选择检测模型'

import torch
import torch.nn as nn
from model.anchor.build_anchor import Anchors
from model.loss.build_loss import build_loss
from model.metrics.build_metrics import build_metrics

class Loss_calculater(nn.Module):
    def __init__(self, cfg) -> None:
        super(Loss_calculater, self).__init__()
        self.img_shape = cfg.TRAIN['TRAIN_IMG_SIZE']

        self.anchors = Anchors()
        # self.all_anchors = self.anchors(torch.zeros(size=(self.batch_size, 3, cfg.TRAIN['TRAIN_IMG_SIZE'],cfg.TRAIN['TRAIN_IMG_SIZE']),
        #                                 dtype=torch.double).cuda())
        self.label_assign = build_metrics(cfg, cfg.MODEL['metrics'])
        
        self.loss = build_loss(cfg.MODEL['loss'], cfg)
        
    def forward(self, imgs, features, targets=None):
        """
        Arguments:
            features (list[Tensor] or ImageList): features from fpn. [large, medium, small]
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the loss of model.
        """

        # anchors = self.anchors(image = images, only_anchors = True)
        anchors = self.anchors(imgs)
        proposals_reg, proposals_cls = features
        proposals_reg = self.flatten_features(proposals_reg)
        proposals_cls = self.flatten_features(proposals_cls)
        anchors = self.flatten_features(anchors)
        cls_pred, reg_pred, obj_pred, cls_target, reg_target, obj_target = \
                            self.label_assign(anchors, targets, proposals_reg, proposals_cls)

        losses, losses_reg, losses_obj, losses_cls = \
                            self.loss(cls_pred, reg_pred, obj_pred, cls_target, reg_target, obj_target) # reg_loss, cls_loss, conf_loss

        return losses, losses_reg, losses_obj, losses_cls

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