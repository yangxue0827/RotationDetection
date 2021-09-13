# -*- coding: utf-8 -*-
# Author:  Qi Ming <chaser.ming@gmail.com>
#          Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>
#
# License: Apache-2.0 license

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment

from libs.utils import bbox_transform
from libs.models.losses.losses import Loss
from utils.order_points import re_order


class LossRIDet(Loss):

    def smooth_l1_loss_quad(self, targets, preds, sigma=3.0):
        sigma_squared = sigma ** 2

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
        return regression_loss

    def linear_sum_assignment_np(self, losses):

        indices = []
        for cnt, loss in enumerate(losses):
            # loss [4, 4], row_ind [4, ], col_ind [4, ]
            row_ind, col_ind = linear_sum_assignment(loss)
            # [4, 3]
            indices.append(np.concatenate(
                [np.ones_like(row_ind.reshape([-1, 1])) * cnt, row_ind.reshape([-1, 1]), col_ind.reshape([-1, 1])],
                axis=1))
        # [-1, 4, 3]
        return np.array(indices, np.int32)

    def hungarian_loss_quad(self, targets, preds, anchor_state, anchors):
        targets = tf.reshape(targets[:, :-1], [-1, 8])

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

        preds = tf.reshape(preds, [-1, 4, 2])
        targets = tf.reshape(targets, [-1, 4, 2])
        # [-1, 4, 4]
        cost_list = [tf.reduce_sum(self.smooth_l1_loss_quad(preds, tf.tile(tf.expand_dims(targets[:, i, :], axis=1), [1, 4, 1])), axis=2) for i in range(4)]
        cost = tf.concat(cost_list, axis=1)
        cost = tf.reshape(cost, [-1, 4, 4])

        indices = tf.py_func(self.linear_sum_assignment_np, inp=[cost], Tout=tf.int32)
        indices = tf.reshape(indices, [-1, 4, 3])
        loss = tf.reduce_sum(tf.gather_nd(cost, indices), axis=1)

        # prepare for normalization
        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        loss = tf.reduce_sum(loss) / normalizer

        return loss

    def hungarian_loss_arbitrary_shaped(self, targets, preds, anchor_state, anchors):
        targets = tf.reshape(targets[:, :-1], [-1, self.cfgs.POINT_SAMPLING_NUM * 2])

        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
        preds = tf.gather(preds, indices)
        targets = tf.gather(targets, indices)
        anchors = tf.gather(anchors, indices)

        # targets = tf.py_func(box_sample.rbox_border_sample,
        #                      inp=[targets],
        #                      Tout=tf.float32)

        # targets = tf.py_func(mask_sample.mask_sampling,
        #                      inp=[tf.reshape(targets, [-1, 4, 2]), self.cfgs.POINT_SAMPLING_NUM],
        #                      Tout=tf.float32)

        # change from delta to abslote data
        if self.cfgs.METHOD == 'H':
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            w = anchors[:, 2] - anchors[:, 0] + 1
            h = anchors[:, 3] - anchors[:, 1] + 1
            # theta = -90 * tf.ones_like(x_c)
            anchors = tf.transpose(tf.stack([x_c, y_c, w, h]))

        preds = bbox_transform.poly_transform_inv(boxes=anchors, deltas=preds, point_num=self.cfgs.POINT_SAMPLING_NUM)

        preds = tf.reshape(preds, [-1, self.cfgs.POINT_SAMPLING_NUM, 2])
        targets = tf.reshape(targets, [-1, self.cfgs.POINT_SAMPLING_NUM, 2])
        # [-1, self.cfgs.POINT_SAMPLING_NUM, self.cfgs.POINT_SAMPLING_NUM]
        cost_list = [tf.reduce_sum(self.smooth_l1_loss_quad(preds, tf.tile(tf.expand_dims(targets[:, i, :], axis=1),
                                                                           [1, self.cfgs.POINT_SAMPLING_NUM, 1])),
                                   axis=2) for i in range(self.cfgs.POINT_SAMPLING_NUM)]

        cost = tf.concat(cost_list, axis=1)
        cost = tf.reshape(cost, [-1, self.cfgs.POINT_SAMPLING_NUM, self.cfgs.POINT_SAMPLING_NUM])

        indices = tf.py_func(self.linear_sum_assignment_np, inp=[cost], Tout=tf.int32)
        indices = tf.reshape(indices, [-1, self.cfgs.POINT_SAMPLING_NUM, 3])
        loss = tf.reduce_sum(tf.gather_nd(cost, indices), axis=1)

        # prepare for normalization
        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        loss = tf.reduce_sum(loss) / normalizer

        return loss

