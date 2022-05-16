import torch
from model.utils import bbox2delta
from .sample_results import SampleResult

class Retinanet_sampler:
    def __init__(self, num_classes):
        self.bbox_coder = bbox2delta
        self.num_calsses = num_classes
        
    def sample(self, assign_result):
        pos_inds = (
            torch.nonzero(assign_result.assigned_gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.assigned_gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        
        bbox_targets = torch.zeros_like(assign_result.bboxes)
        bbox_targets_weights = torch.zeros_like(assign_result.bboxes)
        
        bbox_labels = bbox_targets.new_full((assign_result.num_bboxes, self.num_calsses), 0)
        bbox_labels_weights = torch.zeros_like(bbox_labels)
        
        pos_bbox_targets = assign_result.targets[assign_result.assigned_gt_inds[pos_inds] - 1, :4]
        pos_bbox = assign_result.bboxes[pos_inds]
        pos_bbox_targets = self.bbox_coder(pos_bbox, pos_bbox_targets)
        
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_targets_weights [pos_inds, :] = 1.0
        
        bbox_labels[pos_inds, assign_result.assigned_labels[pos_inds]] = 1
        bbox_labels_weights[pos_inds, :] = 1.0
        bbox_labels_weights[neg_inds, :] = 1.0


        return SampleResult(bbox_targets, bbox_targets_weights, bbox_labels, 
                            bbox_labels_weights, pos_inds, neg_inds)