import torch
import torch.nn as nn

from utils.tools import iou_xyxy_torch
class ATSS(nn.Module):
    """New matcher for anchor-based model, whose full name is: Adaptive Training Sample Selection.

    Refer to <Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection>.
    Refer to mmdetection:
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

    ATSSMatcher must cooperate with 'Centerness' Metric.

    Args:
        cfg (easydict): config for Matcher.
        num_fg_classes (int): number of all foreground classes(do not include 'background').

        In order to be compatible with Class-Specific-Regression for multi-class(to detect overlapped objects),
        Matcher will compute match_result for every fg-class.
    """

    def __init__(self, cfg, num_fg_classes):
        super(ATSS, self).__init__()
        self.negative_label = 0
        self.ignore_label = -1
        self.positive_label = 1
        self.num_fg_classes = num_fg_classes
        self.ignore_cross_boundary = cfg.IGNORE_CROSS_BOUNDARY
        self.integrity_thresh = cfg.INTEGRITY_THRESH
        self.regression_type = cfg.REGRESSION_TYPE
        self.topk = cfg.TOPK

    def forward(self, centerness_matrix, targets, anchors):
        """
        Args:
            centerness_matrix (torch.Tensor): An tensor with shape (N, M), N anchors and M GT_boxes
            target (torch.Tensor) [B, M, 4]: contain GT boxes.
            anchor (torch.Tensor) [B, N, 4]: anchors.

        Returns:
            matches (torch.Tensor): a matrix with shape (M, self.num_fg_classes), where matches[i, j] is a matched
                ground-truth index i in [0, N) for class j.
            match_labels (torch.Tensor): a matrix of shape (M, self.num_fg_classes), where match_labels[i, j] indicates
                whether a prediction i is a true or false positive or ignored for class j.
        """
        anchor_boxes = anchors
        # get mask of invisible anchor
        # if self.regression_type == 'anchor_based':
        #     ignore_anchor_mask = ~anchor.get_field("visibility").all(dim=1)
        # else:
        #     src_w = target.size[0]
        #     src_h = target.size[1]
        #     anchor_center = torch.mean(anchor_boxes.view(anchor_boxes.size(0), -1, 2), dim=1)
        #     invisible_anchor_mask = (anchor_center[:, 0] > src_w) | (anchor_center[:, 1] > src_h)
        #     ignore_anchor_mask = invisible_anchor_mask

        default_matches = centerness_matrix.new_full((centerness_matrix.size(1),), 0, dtype=torch.int64)
        default_match_labels = centerness_matrix.new_full((centerness_matrix.size(1),), self.negative_label, dtype=torch.int8)

        # compute iou matrix between N gt_boxes and M anchor_boxes, shape (N, M)
        iou = iou_xyxy_torch(anchor_boxes, targets)
        # stride for each anchor box, shape (M,)
        anchor_stride = anchor.get_field('feature_stride').clone()
        anchor_stride_unique, num_level_bboxes = torch.unique(anchor_stride, sorted=True, return_counts=True)

        gt_label = target.get_field('labels')
        matches = []
        match_labels = []
        for class_idx in range(self.num_fg_classes):
            class_mask = gt_label == class_idx
            if not torch.any(class_mask):
                matches.append(default_matches)
                match_labels.append(default_match_labels)
            else:
                class_label_idxs = torch.where(class_mask)[0]
                sub_centerness_matrix = centerness_matrix[class_label_idxs]
                sub_iou_matrix = iou_matrix[class_label_idxs]
                num_gt = sub_iou_matrix.size(0)

                candidate_idxs = []
                start_idx = 0
                for bboxes_per_level in num_level_bboxes:
                    # on each pyramid level, for each gt,
                    # select k bbox whose center are closest to the gt center
                    end_idx = start_idx + bboxes_per_level
                    tmp_centerness_matrix = sub_centerness_matrix[:, start_idx:end_idx]
                    _ , topk_idxs_per_level = tmp_centerness_matrix.topk(k=min(self.topk,
                                                                         bboxes_per_level), dim=1, largest=True)
                    candidate_idxs.append(topk_idxs_per_level + start_idx)
                    start_idx = end_idx
                candidate_idxs = torch.cat(candidate_idxs, dim=1)
                tmp_idxs = torch.arange(num_gt).reshape(-1, 1)
                candidate_ious = sub_iou_matrix[tmp_idxs, candidate_idxs]
                candidate_centerness = sub_centerness_matrix[tmp_idxs, candidate_idxs]
                iou_mean_per_gt = candidate_ious.mean(dim=1)
                iou_std_per_gt = candidate_ious.std(dim=1)
                iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
                iou_mask = candidate_ious > iou_thresh_per_gt[:, None]
                # limit the positive anchor's center in gt
                centerness_mask = candidate_centerness > 0
                pos_mask = iou_mask & centerness_mask

                # TODO: if an anchor box is assigned to multiple gts, choose the one with the highest IoU ?
                tmp_mask = torch.zeros_like(pos_mask, dtype=torch.float32)
                tmp_mask[pos_mask] = 1.1
                sub_centerness_matrix[tmp_idxs, candidate_idxs] = tmp_mask
                max_matched_vals, max_matche_labels = sub_centerness_matrix.max(dim=0)
                sub_match_labels = max_matche_labels.new_full(max_matche_labels.size(),
                                                              self.negative_label,
                                                              dtype=torch.int8)
                postive_mask = max_matched_vals == 1.1
                sub_match_labels[postive_mask] = self.positive_label
                # ignore anchor points which are cross-boundary or are inside ignored_gt_boxes.
                sub_match_labels[ignore_anchor_mask] = self.ignore_label

                matches.append(class_label_idxs[max_matche_labels])
                match_labels.append(sub_match_labels)

        matches = torch.stack(matches, dim=1)
        match_labels = torch.stack(match_labels, dim=1)

        return matches, match_labels

