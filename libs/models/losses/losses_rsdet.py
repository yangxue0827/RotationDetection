# -*- coding: utf-8 -*-
# Author:  Wen Qian <qianwen2018@ia.ac.cn>
#          Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>
#
# License: Apache-2.0 license

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from libs.utils import bbox_transform
from libs.models.losses.losses import Loss
from utils.order_points import re_order


class LossRSDet(Loss):

    def modulated_rotation_5p_loss(self, targets, preds, anchor_state, ratios, sigma=3.0):

        targets = tf.reshape(targets, [-1, 5])

        sigma_squared = sigma ** 2
        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
        preds = tf.gather(preds, indices)
        targets = tf.gather(targets, indices)
        ratios = tf.gather(ratios, indices)

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        regression_diff = preds - targets
        regression_diff = tf.abs(regression_diff)
        loss1 = tf.where(
            tf.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        loss1 = tf.reduce_sum(loss1, 1)

        loss2_1 = preds[:, 0] - targets[:, 0]
        loss2_2 = preds[:, 1] - targets[:, 1]
        # loss2_3 = preds[:, 2] - targets[:, 3] - tf.log(ratios)
        # loss2_4 = preds[:, 3] - targets[:, 2] + tf.log(ratios)
        loss2_3 = preds[:, 2] - targets[:, 3] + tf.log(ratios)
        loss2_4 = preds[:, 3] - targets[:, 2] - tf.log(ratios)
        loss2_5 = tf.minimum((preds[:, 4] - targets[:, 4] + 1.570796), (targets[:, 4] - preds[:, 4] + 1.570796))

        box_diff_2 = tf.stack([loss2_1, loss2_2, loss2_3, loss2_4, loss2_5], 1)
        abs_box_diff_2 = tf.abs(box_diff_2)
        loss2 = tf.where(
            tf.less(abs_box_diff_2, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(abs_box_diff_2, 2),
            abs_box_diff_2 - 0.5 / sigma_squared
        )
        loss2 = tf.reduce_sum(loss2, 1)
        loss = tf.minimum(loss1, loss2)
        loss = tf.reduce_sum(loss) / normalizer

        return loss

    def modulated_rotation_8p_loss(self, targets, preds, anchor_state, anchors, sigma=3.0):
        targets = tf.reshape(targets[:, :-1], [-1, 8])

        sigma_squared = sigma ** 2
        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
        preds = tf.gather(preds, indices)
        targets = tf.gather(targets, indices)
        anchors = tf.gather(anchors, indices)

        # change from delta to abslote data
        if self.cfgs.METHOD == 'H':
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            w = anchors[:, 2] - anchors[:, 0] + 1
            h = anchors[:, 3] - anchors[:, 1] + 1
            # theta = -90 * tf.ones_like(x_c)
            anchors = tf.transpose(tf.stack([x_c, y_c, w, h]))

        preds = bbox_transform.qbbox_transform_inv(boxes=anchors, deltas=preds)

        targets = tf.py_func(func=re_order,
                             inp=[targets],
                             Tout=[tf.float32])
        targets = tf.reshape(targets, [-1, 8])

        # prepare for normalization
        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        # loss1
        loss1_1 = (preds[:, 0] - targets[:, 0]) / anchors[:, 2]
        loss1_2 = (preds[:, 1] - targets[:, 1]) / anchors[:, 3]
        loss1_3 = (preds[:, 2] - targets[:, 2]) / anchors[:, 2]
        loss1_4 = (preds[:, 3] - targets[:, 3]) / anchors[:, 3]
        loss1_5 = (preds[:, 4] - targets[:, 4]) / anchors[:, 2]
        loss1_6 = (preds[:, 5] - targets[:, 5]) / anchors[:, 3]
        loss1_7 = (preds[:, 6] - targets[:, 6]) / anchors[:, 2]
        loss1_8 = (preds[:, 7] - targets[:, 7]) / anchors[:, 3]
        box_diff_1 = tf.stack([loss1_1, loss1_2, loss1_3, loss1_4, loss1_5, loss1_6, loss1_7, loss1_8], 1)
        box_diff_1 = tf.abs(box_diff_1)
        loss_1 = tf.where(
            tf.less(box_diff_1, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(box_diff_1, 2),
            box_diff_1 - 0.5 / sigma_squared
        )
        loss_1 = tf.reduce_sum(loss_1, 1)

        # loss2
        loss2_1 = (preds[:, 0] - targets[:, 2]) / anchors[:, 2]
        loss2_2 = (preds[:, 1] - targets[:, 3]) / anchors[:, 3]
        loss2_3 = (preds[:, 2] - targets[:, 4]) / anchors[:, 2]
        loss2_4 = (preds[:, 3] - targets[:, 5]) / anchors[:, 3]
        loss2_5 = (preds[:, 4] - targets[:, 6]) / anchors[:, 2]
        loss2_6 = (preds[:, 5] - targets[:, 7]) / anchors[:, 3]
        loss2_7 = (preds[:, 6] - targets[:, 0]) / anchors[:, 2]
        loss2_8 = (preds[:, 7] - targets[:, 1]) / anchors[:, 3]
        box_diff_2 = tf.stack([loss2_1, loss2_2, loss2_3, loss2_4, loss2_5, loss2_6, loss2_7, loss2_8], 1)
        box_diff_2 = tf.abs(box_diff_2)
        loss_2 = tf.where(
            tf.less(box_diff_2, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(box_diff_2, 2),
            box_diff_2 - 0.5 / sigma_squared
        )
        loss_2 = tf.reduce_sum(loss_2, 1)

        # loss3
        loss3_1 = (preds[:, 0] - targets[:, 6]) / anchors[:, 2]
        loss3_2 = (preds[:, 1] - targets[:, 7]) / anchors[:, 3]
        loss3_3 = (preds[:, 2] - targets[:, 0]) / anchors[:, 2]
        loss3_4 = (preds[:, 3] - targets[:, 1]) / anchors[:, 3]
        loss3_5 = (preds[:, 4] - targets[:, 2]) / anchors[:, 2]
        loss3_6 = (preds[:, 5] - targets[:, 3]) / anchors[:, 3]
        loss3_7 = (preds[:, 6] - targets[:, 4]) / anchors[:, 2]
        loss3_8 = (preds[:, 7] - targets[:, 5]) / anchors[:, 3]
        box_diff_3 = tf.stack([loss3_1, loss3_2, loss3_3, loss3_4, loss3_5, loss3_6, loss3_7, loss3_8], 1)
        box_diff_3 = tf.abs(box_diff_3)
        loss_3 = tf.where(
            tf.less(box_diff_3, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(box_diff_3, 2),
            box_diff_3 - 0.5 / sigma_squared
        )
        loss_3 = tf.reduce_sum(loss_3, 1)

        loss = tf.minimum(tf.minimum(loss_1, loss_2), loss_3)
        loss = tf.reduce_sum(loss) / normalizer

        return loss
