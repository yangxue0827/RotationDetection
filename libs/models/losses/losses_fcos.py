# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2

from utils.quad2rbox import quad2rbox_tf
from libs.models.losses.losses import Loss


class LossFCOS(Loss):

    def focal_loss_fcos(self, pred, labels, alpha=0.25, gamma=2.0):

        # compute the focal loss
        labels = tf.cast(labels, tf.int32)
        labels = tf.one_hot(labels, self.cfgs.CLASS_NUM + 1, axis=-1)
        labels = labels[:, :, 1:]
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
        normalizer = tf.stop_gradient(tf.where(tf.greater(labels, 0)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(focal_cross_entropy_loss) / normalizer

    def smooth_l1_loss_fcos(self, targets, preds, cls_gt, background=0, weight=None, sigma=3.0):

        cls_gt = tf.reshape(cls_gt, [-1, ])
        weight = tf.reshape(weight, [-1, ])
        mask = 1 - tf.cast(tf.equal(cls_gt, background), tf.int32)
        targets = tf.reshape(targets, [-1, 8])
        preds = tf.reshape(preds, [-1, 8])

        sigma_squared = sigma ** 2

        # prepare for normalization
        normalizer = tf.stop_gradient(tf.where(tf.equal(mask, 0)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        loss_tmp = []
        for i in range(4):
            loss_tmp.append((preds[:, i * 2] - targets[:, i * 2]) / 16.0)
            loss_tmp.append((preds[:, i * 2 + 1] - targets[:, i * 2 + 1]) / 16.0)
        box_diff = tf.stack(loss_tmp, 1)
        box_diff = tf.abs(box_diff)
        loss = tf.where(
            tf.less(box_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(box_diff, 2),
            box_diff - 0.5 / sigma_squared
        )

        loss = tf.reduce_sum(loss, 1) * tf.cast(mask, tf.float32)

        loss = tf.reduce_sum(loss) / normalizer

        return loss

    def centerness_loss(self, pred, label, cls_gt, background=0):
        mask = 1 - tf.cast(tf.equal(cls_gt, background), tf.int32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label) * tf.cast(mask, tf.float32)
        # not_neg_mask = tf.cast(tf.greater_equal(pred, 0), tf.float32)
        # loss = (pred * not_neg_mask - pred * label + tf.log(1 + tf.exp(-tf.abs(pred)))) * tf.cast(mask, tf.float32)
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1)

    def smooth_l1_loss_rbox_fcos(self, targets, preds, cls_gt, param_num, background=0, sigma=3.0, weight=None):

        cls_gt = tf.reshape(cls_gt, [-1, ])
        weight = tf.reshape(weight, [-1, ])
        mask = 1 - tf.cast(tf.equal(cls_gt, background), tf.int32)
        targets = tf.reshape(targets, [-1, param_num])
        preds = tf.reshape(preds, [-1, param_num])

        sigma_squared = sigma ** 2

        # prepare for normalization
        normalizer = tf.stop_gradient(tf.where(tf.greater(mask, 0)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        regression_diff = preds - targets
        regression_diff = tf.abs(regression_diff)

        loss = tf.where(
            tf.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        loss = tf.reduce_sum(loss, 1) * tf.cast(mask, tf.float32)
        loss = tf.reduce_sum(loss) / normalizer

        return tf.reduce_sum(loss) / normalizer

    def iou_loss(self, pred, gt, cls_gt, background=0, weight=None):
        mask = 1 - tf.cast(tf.equal(cls_gt, background), tf.int32)

        area_gt = tf.abs(gt[:, :, 2] + gt[:, :, 0]) * tf.abs(gt[:, :, 3] + gt[:, :, 1])
        area_pred = tf.abs(pred[:, :, 2] + pred[:, :, 0]) * tf.abs(pred[:, :, 3] + pred[:, :, 1])

        iw = tf.minimum(pred[:, :, 2], gt[:, :, 2]) + tf.minimum(pred[:, :, 0], gt[:, :, 0])
        ih = tf.minimum(pred[:, :, 3], gt[:, :, 3]) + tf.minimum(pred[:, :, 1], gt[:, :, 1])
        inter = tf.maximum(iw, 0) * tf.maximum(ih, 0)

        union = area_gt + area_pred - inter
        iou = tf.maximum(inter / union, 0)

        # l_theta = 1 - tf.cos(gt[:, :, -1] - pred[:, :, -1])
        l_sin_theta = tf.abs(pred[:, :, -2] - tf.sin(gt[:, :, -1]))
        l_cos_theta = tf.abs(pred[:, :, -1] - tf.cos(gt[:, :, -1]))

        if weight is not None:
            iou *= weight
            l_sin_theta *= weight
            l_cos_theta *= weight
        l_iou = - tf.log(iou + self.cfgs.EPSILON) * tf.cast(mask, tf.float32)
        loss = l_iou + l_sin_theta + l_cos_theta

        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)

