from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from libs.models.samplers.samper import Sampler
from libs.utils.rbbox_overlaps import rbbx_overlaps
from libs.utils import bbox_transform


class RefineAnchorSamplerR3Det(Sampler):

    def refine_anchor_target_layer(self, gt_boxes_r, anchors, pos_threshold, neg_threshold, gpu_id=0):

        anchor_states = np.zeros((anchors.shape[0],))
        labels = np.zeros((anchors.shape[0], self.cfgs.CLASS_NUM))
        if gt_boxes_r.shape[0]:
            # [N, M]

            # overlaps = get_iou_matrix(np.ascontiguousarray(anchors, dtype=np.float32),
            #                           np.ascontiguousarray(gt_boxes_r[:, :-1], dtype=np.float32))
            #
            overlaps = rbbx_overlaps(np.ascontiguousarray(anchors, dtype=np.float32),
                                     np.ascontiguousarray(gt_boxes_r[:, :-1], dtype=np.float32), gpu_id)

            argmax_overlaps_inds = np.argmax(overlaps, axis=1)
            max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

            # compute box regression targets
            target_boxes = gt_boxes_r[argmax_overlaps_inds]

            positive_indices = max_overlaps >= pos_threshold
            ignore_indices = (max_overlaps > neg_threshold) & ~positive_indices

            anchor_states[ignore_indices] = -1
            anchor_states[positive_indices] = 1

            # compute target class labels
            labels[positive_indices, target_boxes[positive_indices, -1].astype(int) - 1] = 1
        else:
            # no annotations? then everything is background
            target_boxes = np.zeros((anchors.shape[0], gt_boxes_r.shape[1]))

        target_delta = bbox_transform.rbbox_transform(ex_rois=anchors, gt_rois=target_boxes,
                                                      scale_factors=self.cfgs.ANCHOR_SCALE_FACTORS)

        return np.array(labels, np.float32), np.array(target_delta, np.float32), \
               np.array(anchor_states, np.float32), np.array(target_boxes, np.float32)




