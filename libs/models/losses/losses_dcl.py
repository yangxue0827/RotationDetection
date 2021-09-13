# -*- coding: utf-8 -*-
# Author: Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>
#
# License: Apache-2.0 license

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from utils.densely_coded_label import angle_label_decode
from libs.models.losses.losses_csl import LossCSL


class LossDCL(LossCSL):

    def angle_cls_period_focal_loss(self, labels, pred, anchor_state, target_boxes, alpha=None, gamma=2.0,
                                    decimal_weight=False, aspect_ratio_threshold=1.5):

        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
        labels = tf.gather(labels, indices)
        pred = tf.gather(pred, indices)
        target_boxes = tf.gather(target_boxes, indices)
        anchor_state = tf.gather(anchor_state, indices)

        # compute the focal loss
        per_entry_cross_ent = - labels * tf.log(tf.sigmoid(pred) + self.cfgs.EPSILON) \
                              - (1 - labels) * tf.log(1 - tf.sigmoid(pred) + self.cfgs.EPSILON)

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

        if decimal_weight:
            angle_decode_labels = tf.py_func(func=angle_label_decode,
                                             inp=[labels, self.cfgs.ANGLE_RANGE, self.cfgs.OMEGA, self.cfgs.ANGLE_MODE],
                                             Tout=[tf.float32])
            angle_decode_labels = tf.reshape(angle_decode_labels, [-1, ]) * -1

            angle_decode_pred = tf.py_func(func=angle_label_decode,
                                           inp=[tf.sigmoid(pred), self.cfgs.ANGLE_RANGE, self.cfgs.OMEGA, self.cfgs.ANGLE_MODE],
                                           Tout=[tf.float32])

            angle_decode_pred = tf.reshape(angle_decode_pred, [-1, ]) * -1

            target_boxes = tf.reshape(target_boxes[:, :-1], [-1, 5])
            x, y, h, w, theta = tf.unstack(target_boxes, axis=-1)
            aspect_ratio = h / w
            period_weight_90 = tf.cast(tf.less_equal(aspect_ratio, aspect_ratio_threshold),
                                       tf.int32) * 2 * 180 / self.cfgs.ANGLE_RANGE
            period_weight_180 = tf.cast(tf.greater(aspect_ratio, aspect_ratio_threshold),
                                        tf.int32) * 1 * 180 / self.cfgs.ANGLE_RANGE

            period_weight = tf.cast(period_weight_90 + period_weight_180, tf.float32)
            diff_weight = tf.reshape(tf.abs(tf.sin(period_weight * (angle_decode_labels - angle_decode_pred))), [-1, 1])

        else:
            diff_weight = tf.ones_like(tf.reshape(anchor_state, [-1, 1]))

        focal_cross_entropy_loss = (diff_weight * modulating_factor * alpha_weight_factor *
                                    per_entry_cross_ent)

        # compute the normalizer: the number of positive anchors
        # normalizer = tf.stop_gradient(tf.where(tf.greater(anchor_state, -2)))
        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
        # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

        return tf.reduce_sum(focal_cross_entropy_loss) / normalizer

    def delta_angle_loss(self, encode_label, target_boxes, preds, anchor_state, sigma=3.0, weight=None):
        sigma_squared = sigma ** 2
        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
        preds = tf.gather(preds, indices)
        target_boxes = tf.gather(target_boxes, indices)
        encode_label = tf.gather(encode_label, indices)

        angle_decode = tf.py_func(angle_label_decode,
                                  inp=[encode_label, self.cfgs.ANGLE_RANGE,
                                       self.cfgs.OMEGA, self.cfgs.ANGLE_MODE],
                                  Tout=[tf.float32])
        angle_decode = tf.reshape(angle_decode, [-1, ]) * -1
        gt_angle = tf.reshape(target_boxes[:, -2], [-1, ])

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        targets = (gt_angle - angle_decode) * (3.1415926 / 180)
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

