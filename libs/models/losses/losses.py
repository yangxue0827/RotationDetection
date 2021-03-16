# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from libs.utils import bbox_transform
from libs.utils.iou_rotate import iou_rotate_calculate2


class Loss(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    # -------------------------------------- single stage methods---------------------------------------
    def focal_loss(self, labels, pred, anchor_state, alpha=0.25, gamma=2.0):

        # filter out "ignore" anchors
        indices = tf.reshape(tf.where(tf.not_equal(anchor_state, -1)), [-1, ])
        labels = tf.gather(labels, indices)
        pred = tf.gather(pred, indices)

        # compute the focal loss
        per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=pred))
        prediction_probabilities = tf.sigmoid(pred)
        p_t = ((labels * prediction_probabilities) +
               ((1 - labels) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if gamma:
            modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_weight_factor = 1.0
        if alpha is not None:
            alpha_weight_factor = (labels * alpha +
                                   (1 - labels) * (1 - alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    per_entry_cross_ent)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        # normalizer = tf.stop_gradient(tf.where(tf.greater_equal(anchor_state, 0)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
        # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

        return tf.reduce_sum(focal_cross_entropy_loss) / normalizer

    def smooth_l1_loss(self, targets, preds, anchor_state, sigma=3.0, weight=None):
        sigma_squared = sigma ** 2
        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
        preds = tf.gather(preds, indices)
        targets = tf.gather(targets, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = preds - targets
        regression_diff = tf.abs(regression_diff)

        regression_loss = tf.where(
            tf.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # regression_loss = tf.reshape(regression_loss, [-1, 5])
        # lx, ly, lh, lw, ltheta = tf.unstack(regression_loss, axis=-1)
        # regression_loss = tf.transpose(tf.stack([lx*1., ly*1., lh*10., lw*1., ltheta*1.]))

        if weight is not None:
            regression_loss = tf.reduce_sum(regression_loss, axis=-1)
            weight = tf.gather(weight, indices)
            regression_loss *= weight

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
        # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

        return tf.reduce_sum(regression_loss) / normalizer

    def iou_smooth_l1_loss_log(self, targets, preds, anchor_state, target_boxes, anchors, sigma=3.0, is_refine=False):
        if self.cfgs.METHOD == 'H' and not is_refine:
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            h = anchors[:, 2] - anchors[:, 0] + 1
            w = anchors[:, 3] - anchors[:, 1] + 1
            theta = -90 * tf.ones_like(x_c)
            anchors = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

        sigma_squared = sigma ** 2
        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])

        preds = tf.gather(preds, indices)
        targets = tf.gather(targets, indices)
        target_boxes = tf.gather(target_boxes, indices)
        anchors = tf.gather(anchors, indices)

        boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors, deltas=preds)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = preds - targets
        regression_diff = tf.abs(regression_diff)
        regression_loss = tf.where(
            tf.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        overlaps = tf.py_func(iou_rotate_calculate2,
                              inp=[tf.reshape(boxes_pred, [-1, 5]), tf.reshape(target_boxes[:, :-1], [-1, 5])],
                              Tout=[tf.float32])

        overlaps = tf.reshape(overlaps, [-1, 1])
        regression_loss = tf.reshape(tf.reduce_sum(regression_loss, axis=1), [-1, 1])
        # -ln(x)
        iou_factor = tf.stop_gradient(-1 * tf.log(overlaps)) / (tf.stop_gradient(regression_loss) + self.cfgs.EPSILON)
        # iou_factor = tf.Print(iou_factor, [iou_factor], 'iou_factor', summarize=50)

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
        # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

        return tf.reduce_sum(regression_loss * iou_factor) / normalizer

    def iou_smooth_l1_loss_exp(self, targets, preds, anchor_state, target_boxes, anchors, sigma=3.0, alpha=1.0, beta=1.0, is_refine=False):
        if self.cfgs.METHOD == 'H' and not is_refine:
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            h = anchors[:, 2] - anchors[:, 0] + 1
            w = anchors[:, 3] - anchors[:, 1] + 1
            theta = -90 * tf.ones_like(x_c)
            anchors = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

        sigma_squared = sigma ** 2
        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])

        preds = tf.gather(preds, indices)
        targets = tf.gather(targets, indices)
        target_boxes = tf.gather(target_boxes, indices)
        anchors = tf.gather(anchors, indices)

        boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors, deltas=preds)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = preds - targets
        regression_diff = tf.abs(regression_diff)
        regression_loss = tf.where(
            tf.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        overlaps = tf.py_func(iou_rotate_calculate2,
                              inp=[tf.reshape(boxes_pred, [-1, 5]), tf.reshape(target_boxes[:, :-1], [-1, 5])],
                              Tout=[tf.float32])

        overlaps = tf.reshape(overlaps, [-1, 1])
        regression_loss = tf.reshape(tf.reduce_sum(regression_loss, axis=1), [-1, 1])
        # 1-exp(1-x)
        iou_factor = tf.stop_gradient(tf.exp(alpha*(1-overlaps)**beta)-1) / (tf.stop_gradient(regression_loss) + self.cfgs.EPSILON)
        # iou_factor = tf.stop_gradient(1-overlaps) / (tf.stop_gradient(regression_loss) + cfgs.EPSILON)
        # iou_factor = tf.Print(iou_factor, [iou_factor], 'iou_factor', summarize=50)

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
        # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

        return tf.reduce_sum(regression_loss * iou_factor) / normalizer

    # -------------------------------------- two stage methods---------------------------------------
    def _smooth_l1_loss_base(self, bbox_pred, bbox_targets, sigma=1.0):
        '''
        :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] in Fast-rcnn
        :param bbox_targets: shape is same as bbox_pred
        :param sigma:
        :return:
        '''
        sigma_2 = sigma ** 2

        box_diff = bbox_pred - bbox_targets

        abs_box_diff = tf.abs(box_diff)

        smoothL1_sign = tf.stop_gradient(
            tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
        loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
                   + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
        return loss_box

    def smooth_l1_loss_rpn(self, bbox_pred, bbox_targets, label, sigma=1.0):
        '''
        :param bbox_pred: [-1, 4]
        :param bbox_targets: [-1, 4]
        :param label: [-1]
        :param sigma:
        :return:
        '''
        value = self._smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
        value = tf.reduce_sum(value, axis=1)  # to sum in axis 1
        rpn_positive = tf.where(tf.greater(label, 0))

        # rpn_select = tf.stop_gradient(rpn_select) # to avoid
        selected_value = tf.gather(value, rpn_positive)
        non_ignored_mask = tf.stop_gradient(
            1.0 - tf.to_float(tf.equal(label, -1)))  # positve is 1.0 others is 0.0

        bbox_loss = tf.reduce_sum(selected_value) / tf.maximum(1.0, tf.reduce_sum(non_ignored_mask))

        return bbox_loss

    def smooth_l1_loss_rcnn(self, bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
        '''
        :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
        :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 4]
        :param label:[-1]
        :param num_classes:
        :param sigma:
        :return:
        '''

        outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

        bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
        bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

        value = self._smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
        value = tf.reduce_sum(value, 2)
        value = tf.reshape(value, [-1, num_classes])

        inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                                 depth=num_classes, axis=1)

        inside_mask = tf.stop_gradient(
            tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

        normalizer = tf.to_float(tf.shape(bbox_pred)[0])
        bbox_loss = tf.reduce_sum(
            tf.reduce_sum(value * inside_mask, 1) * outside_mask) / normalizer

        return bbox_loss

    def smooth_l1_loss_rcnn_h(self, bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
        '''
        :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
        :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 4]
        :param label:[-1]
        :param num_classes:
        :param sigma:
        :return:
        '''

        outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

        bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
        bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

        value = self._smooth_l1_loss_base(bbox_pred,
                                          bbox_targets,
                                          sigma=sigma)
        value = tf.reduce_sum(value, 2)
        value = tf.reshape(value, [-1, num_classes])

        inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                                 depth=num_classes, axis=1)

        inside_mask = tf.stop_gradient(
            tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

        normalizer = tf.to_float(tf.shape(bbox_pred)[0])
        bbox_loss = tf.reduce_sum(
            tf.reduce_sum(value * inside_mask, 1) * outside_mask) / normalizer

        return bbox_loss

    def smooth_l1_loss_rcnn_r(self, bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
        '''
        :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 5]
        :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 5]
        :param label:[-1]
        :param num_classes:
        :param sigma:
        :return:
        '''

        outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

        bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 5])
        bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 5])

        value = self._smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
        value = tf.reduce_sum(value, 2)
        value = tf.reshape(value, [-1, num_classes])

        inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                                 depth=num_classes, axis=1)

        inside_mask = tf.stop_gradient(
            tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

        normalizer = tf.to_float(tf.shape(bbox_pred)[0])
        bbox_loss = tf.reduce_sum(
            tf.reduce_sum(value * inside_mask, 1) * outside_mask) / normalizer

        return bbox_loss

    def iou_smooth_l1_loss_rcnn_r(self, bbox_pred, bbox_targets, label, rois, target_gt_r, num_classes, sigma=1.0):
        '''
        :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 5]
        :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 5]
        :param label:[-1]
        :param num_classes:
        :param sigma:
        :return:
        '''

        outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

        target_gt_r = tf.reshape(tf.tile(tf.reshape(target_gt_r, [-1, 1, 5]), [1, num_classes, 1]), [-1, 5])
        x_c = (rois[:, 2] + rois[:, 0]) / 2
        y_c = (rois[:, 3] + rois[:, 1]) / 2
        h = rois[:, 2] - rois[:, 0] + 1
        w = rois[:, 3] - rois[:, 1] + 1
        theta = -90 * tf.ones_like(x_c)
        rois = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
        rois = tf.reshape(tf.tile(tf.reshape(rois, [-1, 1, 5]), [1, num_classes, 1]), [-1, 5])

        boxes_pred = bbox_transform.rbbox_transform_inv(boxes=rois, deltas=tf.reshape(bbox_pred, [-1, 5]),
                                                        scale_factors=self.cfgs.ROI_SCALE_FACTORS)
        overlaps = tf.py_func(iou_rotate_calculate2,
                              inp=[tf.reshape(boxes_pred, [-1, 5]), tf.reshape(target_gt_r, [-1, 5])],
                              Tout=[tf.float32])
        overlaps = tf.reshape(overlaps, [-1, num_classes])

        bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 5])
        bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 5])

        value = self._smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
        value = tf.reduce_sum(value, 2)
        value = tf.reshape(value, [-1, num_classes])

        inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                                 depth=num_classes, axis=1)

        inside_mask = tf.stop_gradient(
            tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

        iou_factor = tf.stop_gradient(tf.exp((1 - overlaps)) - 1) / (tf.stop_gradient(value) + self.cfgs.EPSILON)

        regression_loss = tf.reduce_sum(value * inside_mask * iou_factor, 1)

        normalizer = tf.to_float(tf.shape(bbox_pred)[0])
        bbox_loss = tf.reduce_sum(regression_loss * outside_mask) / normalizer

        return bbox_loss

    def build_attention_loss(self, mask, featuremap):
        # shape = mask.get_shape().as_list()
        shape = tf.shape(mask)
        featuremap = tf.image.resize_bilinear(featuremap, [shape[0], shape[1]])
        # shape = tf.shape(featuremap)
        # mask = tf.expand_dims(mask, axis=0)
        # mask = tf.image.resize_bilinear(mask, [shape[1], shape[2]])
        # mask = tf.squeeze(mask, axis=0)

        mask = tf.cast(mask, tf.int32)
        mask = tf.reshape(mask, [-1, ])
        featuremap = tf.reshape(featuremap, [-1, 2])
        attention_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=mask, logits=featuremap)
        attention_loss = tf.reduce_mean(attention_loss)
        return attention_loss