# -*- coding: utf-8 -*-
# Author: Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>, <yangxue0827@126.com>
# License: Apache-2.0 license
# Copyright (c) SJTU. ALL Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

from alpharotate.libs.models.losses.losses import Loss
from alpharotate.libs.utils import bbox_transform
from alpharotate.libs.utils.coordinate_convert import coordinate_present_convert


class LossBCD(Loss):

    def get_gaussian_param(self, boxes_pred, target_boxes, shrink_ratio=1.):
        x1, y1, w1, h1, theta1 = tf.unstack(boxes_pred, axis=1)
        x2, y2, w2, h2, theta2 = tf.unstack(target_boxes, axis=1)
        x1 = tf.reshape(x1, [-1, 1])
        y1 = tf.reshape(y1, [-1, 1])
        h1 = tf.reshape(h1, [-1, 1]) * shrink_ratio
        w1 = tf.reshape(w1, [-1, 1]) * shrink_ratio
        theta1 = tf.reshape(theta1, [-1, 1])
        x2 = tf.reshape(x2, [-1, 1])
        y2 = tf.reshape(y2, [-1, 1])
        h2 = tf.reshape(h2, [-1, 1]) * shrink_ratio
        w2 = tf.reshape(w2, [-1, 1]) * shrink_ratio
        theta2 = tf.reshape(theta2, [-1, 1])
        theta1 *= np.pi / 180
        theta2 *= np.pi / 180

        sigma1_1 = w1 / 2 * tf.cos(theta1) ** 2 + h1 / 2 * tf.sin(theta1) ** 2
        sigma1_2 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
        sigma1_3 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
        sigma1_4 = w1 / 2 * tf.sin(theta1) ** 2 + h1 / 2 * tf.cos(theta1) ** 2
        sigma1 = tf.reshape(tf.concat([sigma1_1, sigma1_2, sigma1_3, sigma1_4], axis=-1), [-1, 2, 2]) + tf.linalg.eye(
            2) * 1e-5

        sigma2_1 = w2 / 2 * tf.cos(theta2) ** 2 + h2 / 2 * tf.sin(theta2) ** 2
        sigma2_2 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
        sigma2_3 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
        sigma2_4 = w2 / 2 * tf.sin(theta2) ** 2 + h2 / 2 * tf.cos(theta2) ** 2
        sigma2 = tf.reshape(tf.concat([sigma2_1, sigma2_2, sigma2_3, sigma2_4], axis=-1), [-1, 2, 2]) + tf.linalg.eye(
            2) * 1e-5

        mu1 = tf.reshape(tf.concat([x1, y1], axis=-1), [-1, 1, 2])
        mu2 = tf.reshape(tf.concat([x2, y2], axis=-1), [-1, 1, 2])

        mu1_T = tf.reshape(tf.concat([x1, y1], axis=-1), [-1, 2, 1])
        mu2_T = tf.reshape(tf.concat([x2, y2], axis=-1), [-1, 2, 1])
        return sigma1, sigma2, mu1, mu2, mu1_T, mu2_T

    def bhattacharyya_distance(self, mu1, mu2, mu1_T, mu2_T, sigma1, sigma2):
        sigma1_square = tf.linalg.matmul(sigma1, sigma1)
        sigma2_square = tf.linalg.matmul(sigma2, sigma2)
        sigma_square = 0.5 * (sigma1_square + sigma2_square)
        item1 = 1 / 8 * tf.linalg.matmul(tf.linalg.matmul(mu2 - mu1, tf.linalg.inv(sigma_square)), mu2_T - mu1_T)
        item2 = 1 / 2 * tf.log(tf.linalg.det(sigma_square) / tf.sqrt(tf.linalg.det(tf.linalg.matmul(sigma1_square, sigma2_square))))
        return item1 + item2

    def bhattacharyya_distance_loss(self, preds, anchor_state, target_boxes, anchors, is_refine=False, tau=1.0, func=0, shrink_ratio=1.):
        if self.cfgs.METHOD == 'H' and not is_refine:
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            h = anchors[:, 2] - anchors[:, 0] + 1
            w = anchors[:, 3] - anchors[:, 1] + 1
            theta = -90 * tf.ones_like(x_c)
            anchors = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
        preds = tf.gather(preds, indices)
        target_boxes = tf.gather(target_boxes, indices)
        anchors = tf.gather(anchors, indices)

        target_boxes_ = tf.reshape(target_boxes[:, :-1], [-1, 5])

        if self.cfgs.ANGLE_RANGE == 180:
            anchors = tf.py_func(coordinate_present_convert,
                                 inp=[anchors, -1, False],
                                 Tout=tf.float32)
            anchors = tf.reshape(anchors, [-1, 5])
            target_boxes_ = tf.py_func(coordinate_present_convert,
                                       inp=[target_boxes_, -1, False],
                                       Tout=tf.float32)
            target_boxes_ = tf.reshape(target_boxes_, [-1, 5])

        boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors, deltas=preds)
        boxes_pred = tf.reshape(boxes_pred, [-1, 5])

        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(target_boxes_, boxes_pred, shrink_ratio=shrink_ratio)

        bhattacharyya_distance = tf.reshape(self.bhattacharyya_distance(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        bhattacharyya_distance = tf.maximum(bhattacharyya_distance, 0.0)

        if func == 0:
            bhattacharyya_distance = tf.maximum(tf.sqrt(bhattacharyya_distance), 0.)
        elif func == 1:
            bhattacharyya_distance = tf.maximum(tf.log(bhattacharyya_distance + 1.), 0.)
        else:
            bhattacharyya_distance = tf.maximum(tf.log(tf.sqrt(bhattacharyya_distance) + 1.), 0.)

        if True:
            bhattacharyya_similarity = 1 / (bhattacharyya_distance + tau)
            bhattacharyya_distance_loss = (1 - bhattacharyya_similarity) * 0.05
        else:
            bhattacharyya_distance_loss = bhattacharyya_distance * 0.05

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(bhattacharyya_distance_loss) / normalizer

    def bhattacharyya_distance_loss_two_stage(self, bbox_pred, rois, target_gt, label, num_classes, tau=1.0, func=0):

        one_hot = tf.one_hot(tf.reshape(label, [-1, ]), depth=num_classes, axis=-1)

        indices = tf.reshape(tf.where(tf.greater(label, 0)), [-1, ])
        bbox_pred = tf.gather(tf.reshape(bbox_pred, [-1, num_classes * 5]), indices)
        rois = tf.gather(tf.reshape(rois, [-1, 4]), indices)
        target_gt = tf.gather(tf.reshape(target_gt, [-1, 5]), indices)
        one_hot = tf.gather(tf.reshape(one_hot, [-1, num_classes]), indices)

        bbox_pred = tf.reshape(tf.cast(one_hot, tf.float32), [-1, num_classes, 1]) * tf.reshape(bbox_pred, [-1, num_classes, 5])
        bbox_pred = tf.reduce_sum(bbox_pred, axis=1)

        x_c = (rois[:, 2] + rois[:, 0]) / 2
        y_c = (rois[:, 3] + rois[:, 1]) / 2
        h = rois[:, 2] - rois[:, 0] + 1
        w = rois[:, 3] - rois[:, 1] + 1
        theta = -90 * tf.ones_like(x_c)
        rois = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

        boxes_pred = bbox_transform.rbbox_transform_inv(boxes=rois, deltas=tf.reshape(bbox_pred, [-1, 5]),
                                                        scale_factors=self.cfgs.ROI_SCALE_FACTORS)

        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(boxes_pred, target_gt)

        bhattacharyya_distance = tf.reshape(self.bhattacharyya_distance(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2),
                                            [-1, 1])
        bhattacharyya_distance = tf.maximum(bhattacharyya_distance, 0.0)

        if func == 0:
            bhattacharyya_distance = tf.maximum(tf.sqrt(bhattacharyya_distance), 0.)
        elif func == 1:
            bhattacharyya_distance = tf.maximum(tf.log(bhattacharyya_distance + 1.), 0.)
        else:
            bhattacharyya_distance = tf.maximum(tf.log(tf.sqrt(bhattacharyya_distance) + 1.), 0.)

        if True:
            bhattacharyya_similarity = 1 / (bhattacharyya_distance + tau)
            bhattacharyya_distance_loss = (1 - bhattacharyya_similarity) * 0.05
        else:
            bhattacharyya_distance_loss = bhattacharyya_distance * 0.05

        normalizer = tf.stop_gradient(tf.where(tf.greater(label, 0)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(bhattacharyya_distance_loss) / normalizer
