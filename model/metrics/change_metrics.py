# Modification 2020 RangiLyu
# Copyright 2018-2019 Open-MMLab.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from utils.tools import *
from .assign_results import AssignResult
from model.post_processing import yolo_decode 


class ChangeAssigner(nn.Module):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self, topk):
        super(ChangeAssigner, self).__init__()
        self.topk = topk
    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self, reg_pred, targets, num_level_bboxes, cls_pred, **kwargs):

        target = targets[targets[:, 4] != -1]
        gt_bboxes = target[:, :4]
        gt_labels = target[:, 4]
        
        bboxes = reg_pred[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)


        assigned_gt_inds = bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        max_cls, max_cls_ind = cls_pred.max(1) # [N, 1]
        pos_inds = max_cls > 0.0

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)
        # [N, M]
        distances = (
            (bboxes_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        )
        min_dis, min_dis_ind = distances.min(1)
        pos_label = max_cls_ind == gt_labels[min_dis_ind].long()
        pos_inds = pos_inds & pos_label
        assigned_gt_inds[pos_inds] = min_dis_ind[pos_inds] + 1
        max_overlaps = min_dis

        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1].long()

        
        return AssignResult(
            num_gt, num_bboxes, assigned_gt_inds, max_overlaps, assigned_labels, bboxes, target
        )
