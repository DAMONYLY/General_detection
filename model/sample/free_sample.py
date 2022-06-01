import torch
from model.utils.boxes import bbox2delta
from .sample_results import SampleResult

class Free_sampler:
    
    def __init__(self, bbox_coder, num_classes):
        self.bbox_coder = bbox2delta
        self.num_calsses = num_classes
        
    def sample(self, assign_result, **kwargs):
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
        bbox_objs = bbox_targets.new_full((assign_result.num_bboxes, ), 0)
        bbox_objs_weights = torch.zeros_like(bbox_objs)

        pos_bbox_targets = assign_result.targets[assign_result.assigned_gt_inds[pos_inds] - 1, :4]

        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_targets_weights[pos_inds, :] = 1.0
        
        bbox_labels[pos_inds, assign_result.assigned_labels[pos_inds]] = 1
        bbox_labels_weights[pos_inds, :] = 1.0
        bbox_labels_weights[neg_inds, :] = 1.0

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            # pos_bbox_targets_weights = bbox_targets_weights[pos_inds]
            pos_anchors = assign_result.bboxes[pos_inds]
            bbox_objs[pos_inds] = self.centerness_target(pos_anchors, pos_bbox_targets)
            bbox_objs_weights[pos_inds] = 1


        return SampleResult(bbox_targets, bbox_targets_weights,
                            bbox_labels, bbox_labels_weights, 
                            bbox_objs, bbox_objs_weights,
                            pos_inds, neg_inds)

    def centerness_target(self, anchors, gts):
        # only calculate pos centerness targets, otherwise there may be nan
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness