import torch.nn as nn
import torch
from model.anchor.build_anchor import Anchors
from model.loss.build_loss import build_loss
from model.metrics.build_metrics import build_metrics
from model.sample.build_sample import build_sampler
from model.post_processing.yolo_decoder import yolo_decode
from utils.tools import IOU_xyxy_torch_same

class Loss_calculater(nn.Module):
    """
    Post-processing of the features extracted by the detector.
    Includes label assign and loss calculation.
    """
    def __init__(self, cfg) -> None:
        super(Loss_calculater, self).__init__()
        self.anchors = Anchors(cfg.Model.anchors)
        self.assigner = build_metrics(cfg)
        self.sampler = build_sampler(cfg)
        self.loss = build_loss(cfg.Model.loss)
        self.img_size = cfg.Data.train.pipeline.input_size
        
    def forward(self, features, targets):
        """
        Arguments:
            features (list[Tensor]): features from head [reg, cls]
            targets (Tensor)[B, num_gt, 4]: ground-truth boxes present in the image 
        Returns:
            loss (Tensor): all loss of model.
            loss_reg (Tensor): reg branch loss of model.
            loss_cls (Tensor): cls branch loss of model.
        """

        proposals_reg, proposals_cls = features
        num_level_bboxes = [feature.size(1) for feature in proposals_reg]
        proposals_reg = torch.cat(proposals_reg, dim=1)
        proposals_cls = torch.cat(proposals_cls, dim=1)

        assert proposals_reg.size(0) == proposals_cls.size(0) == targets.size(0)
        batch_size = proposals_reg.size(0)
        
        bboxes = self.anchors(self.img_size, device=proposals_reg.device, dtype=proposals_reg.dtype)
        bboxes = bboxes.unsqueeze(0).repeat(batch_size, 1, 1)
        
        reg_targets = []
        reg_weights = []
        cls_targets = []
        cls_weights = []
        num_pos_inds = 0
        
        for batch in range(batch_size):
            assigned_results = self.assigner.assign(bboxes[batch], targets[batch], num_level_bboxes, proposals_reg[batch])
            
            sampled_results = self.sampler.sample(assigned_results, reg_feature=proposals_reg[batch])
            
            reg_targets.append(sampled_results.bbox_targets)
            reg_weights.append(sampled_results.bbox_targets_weights)
            cls_targets.append(sampled_results.bbox_labels)
            cls_weights.append(sampled_results.bbox_labels_weights)
            num_pos_inds += sampled_results.num_pos_inds
            
        reg_targets = torch.cat(reg_targets)
        reg_weights = torch.cat(reg_weights)
        cls_targets = torch.cat(cls_targets)
        cls_weights = torch.cat(cls_weights)
        reg_pred = proposals_reg.view(-1, proposals_reg.size(-1))
        cls_pred = proposals_cls.view(-1, proposals_cls.size(-1))
        
        losses, losses_reg, losses_cls = self.loss(reg_pred, reg_targets, reg_weights,
                                                   cls_pred, cls_targets, cls_weights,
                                                   num_pos_inds) # reg_loss, cls_loss, conf_loss
        return losses, losses_reg, losses_cls

