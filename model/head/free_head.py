import torch.nn as nn
import torch
from loguru import logger
from model.utils.init_weights import *
from model.layers import ConvModule


from model.anchor import Anchors
from model.loss import build_loss
from model.metrics import build_metrics
from model.sample import build_sampler

class free_Head(nn.Module):
    def __init__(self, num_features_in, cfg):
        super(free_Head, self).__init__()
        
        reg_feature_size = cfg.Model.head.reg_head.get('mid_channel', 256)
        self.reg_out_channel = cfg.Model.head.reg_head.get('out_channel', 4)
        self.reg_stack_layers = cfg.Model.head.reg_head.get('stack_layers', 4)
        
        cls_feature_size = cfg.Model.head.cls_head.get('mid_channel', 256)
        self.cls_out_channel = cfg.Model.head.cls_head.out_channel
        self.cls_stack_layers = cfg.Model.head.cls_head.get('stack_layers', 4)   
        
        self.reg_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.reg_stack_layers):
            reg_channel = num_features_in if i == 0 else reg_feature_size
            self.reg_convs.append(ConvModule(filters_in=reg_channel,
                                             filters_out=reg_feature_size,
                                             kernel_size=3,
                                             stride=1,
                                             pad=1,
                                             norm='bn',
                                             activate='relu'
                                             )
                                  )
        for i in range(self.cls_stack_layers):
            cls_channel = num_features_in if i == 0 else cls_feature_size
            self.cls_convs.append(ConvModule(filters_in=cls_channel,
                                             filters_out=cls_feature_size,
                                             kernel_size=3,
                                             stride=1,
                                             pad=1,
                                             norm='bn',
                                             activate='relu'
                                             )
                                  )
            
        self.reg_head = ConvModule(reg_feature_size, self.reg_out_channel, kernel_size=3, pad=1)
        self.obj_head = ConvModule(reg_feature_size, 1, kernel_size=3, pad=1, activate='sigmoid')
        
        self.cls_head = ConvModule(cls_feature_size, self.cls_out_channel, kernel_size=3, pad=1, activate='sigmoid')
        

        self.anchors = Anchors(cfg.Model.anchors)
        self.assigner = build_metrics(cfg)
        self.sampler = build_sampler(cfg)
        self.loss = build_loss(cfg.Model.loss)
        self.img_size = cfg.Data.train.pipeline.input_size
    
    def init_weights(self):
        logger.info("=> Initialize Head ...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


    def forward(self, x):
        b, _, _, _ = x.shape
        reg_feature = x
        cls_feature = x
        for reg_conv, cls_conv in zip(self.reg_convs, self.cls_convs):
            reg_feature = reg_conv(reg_feature)
            cls_feature = cls_conv(cls_feature)
        reg_out = self.reg_head(reg_feature)
        obj_out = self.obj_head(reg_feature)
        reg_out = torch.cat((reg_out, obj_out), dim=1)
        cls_out = self.cls_head(cls_feature)
        
        reg_out = reg_out.permute(0, 2, 3, 1).contiguous().view(b, -1, self.reg_out_channel+1)
        cls_out = cls_out.permute(0, 2, 3, 1).contiguous().view(b, -1, self.cls_out_channel)
        
        return reg_out, cls_out
    
    def loss_calculater(self, features, targets, input_size):
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
        obj_targets = []
        obj_weights = []
        num_pos_inds = 0
        
        for batch in range(batch_size):
            assigned_results = self.assigner.assign(bboxes[batch], targets[batch], num_level_bboxes, feature=proposals_reg[batch])
             
            sampled_results = self.sampler.sample(assigned_results, reg_feature=proposals_reg[batch])
            
            reg_targets.append(sampled_results.bbox_targets)
            reg_weights.append(sampled_results.bbox_targets_weights)
            cls_targets.append(sampled_results.bbox_labels)
            cls_weights.append(sampled_results.bbox_labels_weights)
            obj_targets.append(sampled_results.bbox_objs)
            obj_weights.append(sampled_results.bbox_objs_weights)
            num_pos_inds += sampled_results.num_pos_inds
            
        reg_targets = torch.cat(reg_targets)
        reg_weights = torch.cat(reg_weights)
        cls_targets = torch.cat(cls_targets)
        cls_weights = torch.cat(cls_weights)
        obj_targets = torch.cat(obj_targets)
        obj_weights = torch.cat(obj_weights)

        reg_pred = proposals_reg.view(-1, proposals_reg.size(-1))
        obj_pred = reg_pred[:, 4]
        reg_pred = reg_pred[:, :4]
        cls_pred = proposals_cls.view(-1, proposals_cls.size(-1))
        
        losses, losses_reg, losses_cls, losses_obj = self.loss(reg_pred, reg_targets, reg_weights,
                                                   cls_pred, cls_targets, cls_weights,
                                                   obj_pred, obj_targets, obj_weights,
                                                   num_pos_inds) # reg_loss, cls_loss, conf_loss
        return {"losses": losses, 
                "losses_reg": losses_reg,
                "losses_cls": losses_cls,
                "losses_obj": losses_obj,
                }
