from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from libs.models.samplers.samper import Sampler
from libs.utils.rbbox_overlaps import rbbx_overlaps
from libs.utils import bbox_transform
from libs.utils.coordinate_convert import coordinate_present_convert


class RefineAnchorSamplerR3DetDCL(Sampler):

    def refine_anchor_target_layer(self, gt_boxes_r, gt_encode_label, anchors, pos_threshold, neg_threshold, gpu_id=0):

        anchor_states = np.zeros((anchors.shape[0],))
        labels = np.zeros((anchors.shape[0], self.cfgs.CLASS_NUM))
        if gt_boxes_r.shape[0]:
            # [N, M]

            # if cfgs.ANGLE_RANGE == 180:
            #     gt_boxes_r_ = coordinate_present_convert(gt_boxes_r[:, :-1], 1)
            #
            #     overlaps = rbbx_overlaps(np.ascontiguousarray(anchors, dtype=np.float32),
            #                              np.ascontiguousarray(gt_boxes_r_, dtype=np.float32), gpu_id)
            # else:
            overlaps = rbbx_overlaps(np.ascontiguousarray(anchors, dtype=np.float32),
                                     np.ascontiguousarray(gt_boxes_r[:, :-1], dtype=np.float32), gpu_id)

            # overlaps = np.clip(overlaps, 0.0, 1.0)

            argmax_overlaps_inds = np.argmax(overlaps, axis=1)
            max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

            # compute box regression targets
            target_boxes = gt_boxes_r[argmax_overlaps_inds]
            target_encode_label = gt_encode_label[argmax_overlaps_inds]

            positive_indices = max_overlaps >= pos_threshold
            ignore_indices = (max_overlaps > neg_threshold) & ~positive_indices
            anchor_states[ignore_indices] = -1
            anchor_states[positive_indices] = 1

            # compute target class labels
            labels[positive_indices, target_boxes[positive_indices, -1].astype(int) - 1] = 1
        else:
            # no annotations? then everything is background
            target_boxes = np.zeros((anchors.shape[0], gt_boxes_r.shape[1]))
            target_encode_label = np.zeros((anchors.shape[0], gt_encode_label.shape[1]))

        if self.cfgs.ANGLE_RANGE == 180:
            anchors = coordinate_present_convert(anchors, mode=-1)
            target_boxes = coordinate_present_convert(target_boxes, mode=-1)

        target_delta = bbox_transform.rbbox_transform(ex_rois=anchors, gt_rois=target_boxes,
                                                      scale_factors=self.cfgs.ANCHOR_SCALE_FACTORS)

        return np.array(labels, np.float32), np.array(target_delta[:, :-1], np.float32), \
               np.array(anchor_states, np.float32), np.array(target_boxes, np.float32), \
               np.array(target_encode_label, np.float32)




