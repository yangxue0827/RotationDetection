from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from libs.models.samplers.samper import Sampler
from libs.utils.cython_utils.cython_bbox import bbox_overlaps
from libs.utils.rbbox_overlaps import rbbx_overlaps
from libs.utils import bbox_transform
from libs.utils.coordinate_convert import coordinate_present_convert


class AnchorSamplerRetinaNet(Sampler):

    def anchor_target_layer(self, gt_boxes_h_batch, gt_boxes_r_batch, anchor_batch, gpu_id=0):

        all_labels, all_target_delta, all_anchor_states, all_target_boxes = [], [], [], []
        for i in range(self.cfgs.BATCH_SIZE):
            anchors = np.array(anchor_batch[i], np.float32)
            gt_boxes_h = gt_boxes_h_batch[i, :, :]
            gt_boxes_r = gt_boxes_r_batch[i, :, :]
            anchor_states = np.zeros((anchors.shape[0],))
            labels = np.zeros((anchors.shape[0], self.cfgs.CLASS_NUM))
            if gt_boxes_r.shape[0]:
                # [N, M]

                if self.cfgs.METHOD == 'H':
                    overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),
                                             np.ascontiguousarray(gt_boxes_h, dtype=np.float))
                else:
                    overlaps = rbbx_overlaps(np.ascontiguousarray(anchors, dtype=np.float32),
                                             np.ascontiguousarray(gt_boxes_r[:, :-1], dtype=np.float32), gpu_id)

                argmax_overlaps_inds = np.argmax(overlaps, axis=1)
                max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

                # compute box regression targets
                target_boxes = gt_boxes_r[argmax_overlaps_inds]

                positive_indices = max_overlaps >= self.cfgs.IOU_POSITIVE_THRESHOLD
                ignore_indices = (max_overlaps > self.cfgs.IOU_NEGATIVE_THRESHOLD) & ~positive_indices

                anchor_states[ignore_indices] = -1
                anchor_states[positive_indices] = 1

                # compute target class labels
                labels[positive_indices, target_boxes[positive_indices, -1].astype(int) - 1] = 1
            else:
                # no annotations? then everything is background
                target_boxes = np.zeros((anchors.shape[0], gt_boxes_r.shape[1]))

            if self.cfgs.METHOD == 'H':
                x_c = (anchors[:, 2] + anchors[:, 0]) / 2
                y_c = (anchors[:, 3] + anchors[:, 1]) / 2
                h = anchors[:, 2] - anchors[:, 0] + 1
                w = anchors[:, 3] - anchors[:, 1] + 1
                theta = -90 * np.ones_like(x_c)
                anchors = np.vstack([x_c, y_c, w, h, theta]).transpose()

            if self.cfgs.ANGLE_RANGE == 180:
                anchors = coordinate_present_convert(anchors, mode=-1)
                target_boxes = coordinate_present_convert(target_boxes, mode=-1)
            target_delta = bbox_transform.rbbox_transform(ex_rois=anchors, gt_rois=target_boxes)

            all_labels.append(labels)
            all_target_delta.append(target_delta)
            all_anchor_states.append(anchor_states)
            all_target_boxes.append(target_boxes)

        return np.array(all_labels, np.float32), np.array(all_target_delta, np.float32), \
               np.array(all_anchor_states, np.float32), np.array(all_target_boxes, np.float32)




