from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from libs.models.samplers.samper import Sampler


class AnchorSamplerATSS(Sampler):

    def bbox_overlap(self, boxes, gt_boxes):
        """Calculates the overlap between proposal and ground truth boxes.
        Some `gt_boxes` may have been padded.  The returned `iou` tensor for these
        boxes will be -1.
        Args:
            boxes: a tensor with a shape of [N, 4]. N is the number of
                proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
                last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
            gt_boxes: a tensor with a shape of [MAX_NUM_INSTANCES, 4]. This
                tensor might have paddings with a negative value.
        Returns:
            iou: a tensor with as a shape of [N, MAX_NUM_INSTANCES].
        """
        with tf.name_scope("bbox_overlap"):
            bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
                value=boxes, num_or_size_splits=4, axis=-1)
            gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
                value=gt_boxes, num_or_size_splits=4, axis=-1)

            # Calculates the intersection area.
            i_xmin = tf.maximum(bb_x_min, tf.transpose(gt_x_min, [1, 0]))
            i_xmax = tf.minimum(bb_x_max, tf.transpose(gt_x_max, [1, 0]))
            i_ymin = tf.maximum(bb_y_min, tf.transpose(gt_y_min, [1, 0]))
            i_ymax = tf.minimum(bb_y_max, tf.transpose(gt_y_max, [1, 0]))
            i_area = tf.maximum((i_xmax - i_xmin), 0) * tf.maximum((i_ymax - i_ymin), 0)

            # Calculates the union area.
            bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
            gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
            # Adds a small epsilon to avoid divide-by-zero.
            u_area = bb_area + tf.transpose(gt_area, [1, 0]) - i_area + 1e-8

            # Calculates IoU.
            iou = i_area / u_area

            # Fills -1 for padded ground truth boxes.
            # padding_mask = tf.less(i_xmin, tf.zeros_like(i_xmin))
            # iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

            return iou

    def rbbox_transform(self, ex_rois, gt_rois, scale_factors=None):
        targets_dx = (gt_rois[:, 0] - ex_rois[:, 0]) / (ex_rois[:, 2] + 1)
        targets_dy = (gt_rois[:, 1] - ex_rois[:, 1]) / (ex_rois[:, 3] + 1)

        targets_dw = tf.log(gt_rois[:, 2] / (ex_rois[:, 2] + 1) + 1e-5)
        targets_dh = tf.log(gt_rois[:, 3] / (ex_rois[:, 3] + 1) + 1e-5)

        targets_dtheta = (gt_rois[:, 4] - ex_rois[:, 4]) * 3.1415916 / 180

        if scale_factors:
            targets_dx *= scale_factors[0]
            targets_dy *= scale_factors[1]
            targets_dw *= scale_factors[2]
            targets_dh *= scale_factors[3]
            targets_dtheta *= scale_factors[4]

        targets = tf.transpose(tf.stack([targets_dx, targets_dy, targets_dw, targets_dh, targets_dtheta]))

        return targets

    def anchor_target_layer(self, gt_boxes_h, gt_boxes_r, anchors, topk=9):
        """Assign gt to bboxes.
            The assignment is done in following steps
            1. compute iou between all proposal and gt_box
            2. compute center distance between all proposal and gt_box
            3. on each pyramid level, for each gt, select k bbox whose center
               are closest to the gt center.
            4. get corresponding iou for the these candidates, and compute the
               mean and std, set mean + std as the iou threshold
            5. select these candidates whose iou are greater than or equal to
               the threshold as postive
            6. limit the positive sample's center in gt_box
           Args:
                proposals (Tensor): Bounding boxes to be assigned, shape (n, 4).
                gt_boxes (Tensor): Ground-truth boxes, shape (k, 4).
                gt_labels (Tensor): Ground-truth labels, shape (k, ).
                gt_boxes_ignore: Ground truth bboxes that are
                    labelled as `ignored`, e.g., crowd boxes in COCO.
            Returns:
                target_boxes, target_labels
        """
        INF = 10e10

        # 1. compute iou between all proposal and gt_box
        overlaps = self.bbox_overlap(gt_boxes_h[:, :-1], anchors)  # [k, n]
        num_gts = tf.shape(overlaps)[0]  # [k, ]
        num_proposals = tf.shape(overlaps)[1]  # [n, ]

        # 2. compute center distance between all proposal and gt_box
        gt_centers = (gt_boxes_h[:, 0:2] + gt_boxes_h[:, 2:4]) * 0.5
        proposal_centers = (anchors[:, 0:2] + anchors[:, 2:4]) * 0.5
        distances = tf.math.sqrt(
            tf.reduce_sum(
                tf.math.squared_difference(
                    gt_centers[:, None, :], proposal_centers[None, :, :]), -1))  # (k, n)

        # 3. on each pyramid level, for each gt, select k bbox whose center
        #    are closest to the gt center.
        _, topk_inds = tf.nn.top_k(-distances, k=topk)  # (k, topk)

        inds = tf.stack([tf.reshape(tf.repeat(tf.range(num_gts), topk), [num_gts, topk]), topk_inds], -1)
        # inds = tf.stack([tf.reshape(tf.tile(tf.reshape(tf.range(num_gts), [num_gts, 1]), [1, topk]), [num_gts, topk]), topk_inds], -1)
        # 4. get corresponding iou for the these candidates, and compute the
        #    mean and std, set mean + std as the iou threshold
        candidate_overlaps = tf.gather_nd(overlaps, inds)  # [k, topk]
        # mean_per_gt, var_per_gt = tf.nn.moments(candidate_overlaps, 1, keepdims=True)
        mean_per_gt, var_per_gt = tf.nn.moments(candidate_overlaps, 1, keep_dims=True)
        std_per_gt = tf.math.sqrt(var_per_gt)
        # mean_per_gt = tf.Print(mean_per_gt, [mean_per_gt], 'mean_per_gt', summarize=100)
        # std_per_gt = tf.Print(std_per_gt, [std_per_gt], 'std_per_gt', summarize=100)
        overlaps_thresh_per_gt = mean_per_gt + std_per_gt

        # 5. select these candidates whose iou are greater than or equal to
        #    the threshold as postive
        is_pos = candidate_overlaps >= overlaps_thresh_per_gt  # (k, topk)
        # 6. limit the positive sample's center in gt_boxes

        # calculate the left, top, right, bottom distance between
        # positive box center and gt_box side
        left = tf.tile(tf.expand_dims(proposal_centers[:, 0], 0), [num_gts, 1]) - gt_boxes_h[:, 0:1]  # (k, n)
        top = tf.tile(tf.expand_dims(proposal_centers[:, 1], 0), [num_gts, 1]) - gt_boxes_h[:, 1:2]
        right = gt_boxes_h[:, 2:3] - tf.tile(tf.expand_dims(proposal_centers[:, 0], 0), [num_gts, 1])
        bottom = gt_boxes_h[:, 3:4] - tf.tile(tf.expand_dims(proposal_centers[:, 1], 0), [num_gts, 1])
        is_in_gt = tf.reduce_min(tf.stack([left, top, right, bottom], -1), -1) > 0.01  # (k, n)
        is_in_gt = tf.gather_nd(is_in_gt, inds)
        is_pos = tf.logical_and(is_pos, is_in_gt)

        topk_inds += (tf.reshape(tf.repeat(tf.range(num_gts), topk), [num_gts, topk]) * num_proposals)
        # topk_inds += (tf.reshape(tf.tile(tf.reshape(tf.range(num_gts), [num_gts, 1]), [1, topk]), [num_gts, topk]) * num_proposals)
        candidate_inds = tf.boolean_mask(topk_inds, is_pos)

        # if an anchor box is assigned to multiple gts
        # the one with highest IoU will be seleted.
        overlaps_inf = tf.cast(tf.fill([num_gts * num_proposals], -INF), tf.float32)

        overlaps_inf = tf.tensor_scatter_nd_update(
            overlaps_inf, candidate_inds[:, None],
            tf.gather(tf.reshape(overlaps, [num_gts * num_proposals]), candidate_inds))

        overlaps_inf = tf.reshape(overlaps_inf, (num_gts, num_proposals))
        max_overlaps = tf.reduce_max(overlaps_inf, 0)
        argmax_overlaps = tf.argmax(overlaps_inf, 0)

        target_boxes = tf.gather(gt_boxes_r, argmax_overlaps)
        target_labels = tf.gather(gt_boxes_h[:, -1], argmax_overlaps)
        positive = max_overlaps > -INF
        target_labels = tf.where(positive, tf.cast(target_labels, tf.int64), tf.zeros_like(positive, tf.int64))
        anchor_states = tf.where(positive, tf.ones_like(positive, tf.int64), tf.zeros_like(positive, tf.int64))

        if self.cfgs.METHOD == 'H':
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            h = anchors[:, 2] - anchors[:, 0] + 1
            w = anchors[:, 3] - anchors[:, 1] + 1
            theta = -90 * tf.ones_like(x_c)
            anchors = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

        target_delta = self.rbbox_transform(ex_rois=anchors, gt_rois=target_boxes)

        return target_labels, target_delta, anchor_states, target_boxes





