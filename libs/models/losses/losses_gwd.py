# --------------------------------------------------------
# gwd tensorflow version

# Author: Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>
#
# License: Apache-2.0 license
# --------------------------------------------------------

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from libs.models.losses.losses import Loss
from libs.utils import bbox_transform
from libs.utils.coordinate_convert import coordinate90_2_180_tf, coordinate_present_convert


class LossGWD(Loss):
    def wasserstein_distance_sigma(self, sigma1, sigma2):
        wasserstein_distance_item2 = tf.linalg.matmul(sigma1, sigma1) + tf.linalg.matmul(sigma2,
                                                                                     sigma2) - 2 * tf.linalg.sqrtm(
            tf.linalg.matmul(tf.linalg.matmul(sigma1, tf.linalg.matmul(sigma2, sigma2)), sigma1))
        wasserstein_distance_item2 = tf.linalg.trace(wasserstein_distance_item2)
        return wasserstein_distance_item2

    def sigma_l2(self, sigma1, sigma2):
        sigma_l2_sum = tf.reduce_mean(tf.reduce_mean(tf.pow(sigma1 - sigma2, 2), axis=-1), axis=-1)
        return sigma_l2_sum

    def wasserstein_distance_loss(self, preds, anchor_state, target_boxes, anchors,
                                  is_refine=False, tau=1.0, func=tf.log):
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

        x1, y1, w1, h1, theta1 = tf.unstack(boxes_pred, axis=1)
        x2, y2, w2, h2, theta2 = tf.unstack(target_boxes_, axis=1)
        x1 = tf.reshape(x1, [-1, 1])
        y1 = tf.reshape(y1, [-1, 1])
        h1 = tf.reshape(h1, [-1, 1])
        w1 = tf.reshape(w1, [-1, 1])
        theta1 = tf.reshape(theta1, [-1, 1])
        x2 = tf.reshape(x2, [-1, 1])
        y2 = tf.reshape(y2, [-1, 1])
        h2 = tf.reshape(h2, [-1, 1])
        w2 = tf.reshape(w2, [-1, 1])
        theta2 = tf.reshape(theta2, [-1, 1])
        theta1 *= np.pi / 180
        theta2 *= np.pi / 180

        sigma1_1 = w1 / 2 * tf.cos(theta1) ** 2 + h1 / 2 * tf.sin(theta1) ** 2
        sigma1_2 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
        sigma1_3 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
        sigma1_4 = w1 / 2 * tf.sin(theta1) ** 2 + h1 / 2 * tf.cos(theta1) ** 2
        sigma1 = tf.reshape(tf.concat([sigma1_1, sigma1_2, sigma1_3, sigma1_4], axis=-1), [-1, 2, 2])

        sigma2_1 = w2 / 2 * tf.cos(theta2) ** 2 + h2 / 2 * tf.sin(theta2) ** 2
        sigma2_2 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
        sigma2_3 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
        sigma2_4 = w2 / 2 * tf.sin(theta2) ** 2 + h2 / 2 * tf.cos(theta2) ** 2
        sigma2 = tf.reshape(tf.concat([sigma2_1, sigma2_2, sigma2_3, sigma2_4], axis=-1), [-1, 2, 2])

        wasserstein_distance_item1 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        wasserstein_distance_item2 = tf.reshape(self.wasserstein_distance_sigma(sigma1, sigma2), [-1, 1])
        # wasserstein_distance_item3 = tf.reshape(self.sigma_l2(sigma1, sigma2), [-1, 1])
        wasserstein_distance = tf.maximum(wasserstein_distance_item1 + wasserstein_distance_item2, 0.0)

        wasserstein_distance = tf.maximum(func(wasserstein_distance + 1e-3), 0.0)

        if True:
            wasserstein_similarity = 1 / (wasserstein_distance + tau)
            wasserstein_loss = 1 - wasserstein_similarity
        else:
            wasserstein_loss = wasserstein_distance * 0.05   # better

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(wasserstein_loss) / normalizer

    def wasserstein_distance_norm_loss(self, preds, anchor_state, target_boxes, anchors, is_refine=False):
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

        x1, y1, w1, h1, theta1 = tf.unstack(boxes_pred, axis=1)
        x2, y2, w2, h2, theta2 = tf.unstack(target_boxes_, axis=1)
        x1 = tf.reshape(x1, [-1, 1])
        y1 = tf.reshape(y1, [-1, 1])
        h1 = tf.reshape(h1, [-1, 1])
        w1 = tf.reshape(w1, [-1, 1])
        theta1 = tf.reshape(theta1, [-1, 1])
        x2 = tf.reshape(x2, [-1, 1])
        y2 = tf.reshape(y2, [-1, 1])
        h2 = tf.reshape(h2, [-1, 1])
        w2 = tf.reshape(w2, [-1, 1])
        theta2 = tf.reshape(theta2, [-1, 1])
        theta1 *= np.pi / 180
        theta2 *= np.pi / 180

        sigma1_1 = w1 / 2 * tf.cos(theta1) ** 2 + h1 / 2 * tf.sin(theta1) ** 2
        sigma1_2 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
        sigma1_3 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
        sigma1_4 = w1 / 2 * tf.sin(theta1) ** 2 + h1 / 2 * tf.cos(theta1) ** 2
        sigma1 = tf.reshape(tf.concat([sigma1_1, sigma1_2, sigma1_3, sigma1_4], axis=-1), [-1, 2, 2])

        sigma2_1 = w2 / 2 * tf.cos(theta2) ** 2 + h2 / 2 * tf.sin(theta2) ** 2
        sigma2_2 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
        sigma2_3 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
        sigma2_4 = w2 / 2 * tf.sin(theta2) ** 2 + h2 / 2 * tf.cos(theta2) ** 2
        sigma2 = tf.reshape(tf.concat([sigma2_1, sigma2_2, sigma2_3, sigma2_4], axis=-1), [-1, 2, 2])

        wasserstein_distance_item1 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        wasserstein_distance_item2 = tf.reshape(self.wasserstein_distance_sigma(sigma1, sigma2), [-1, 1])
        # wasserstein_distance_item3 = tf.reshape(self.sigma_l2(sigma1, sigma2), [-1, 1])
        wasserstein_distance = tf.maximum(wasserstein_distance_item1 + wasserstein_distance_item2, 0.0)

        wasserstein_distance = tf.maximum(tf.sqrt(wasserstein_distance), 0.0)
        wasserstein_distance /= tf.sqrt(tf.sqrt(tf.reshape(w1 * h1 * w2 * h2, [-1, 1])))
        # wasserstein_loss = tf.exp(-1 * wasserstein_distance)

        if True:
            wasserstein_similarity = 1 / (wasserstein_distance + 2.)
            wasserstein_loss = 1 - wasserstein_similarity
        else:
            wasserstein_loss = wasserstein_distance * 0.05   # better

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(wasserstein_loss) / normalizer

    def probiou(self, mode, preds, anchor_state, target_boxes, anchors,
                is_refine=False):
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

        x1, y1, w1, h1, theta1 = tf.unstack(boxes_pred, axis=1)
        x2, y2, w2, h2, theta2 = tf.unstack(target_boxes_, axis=1)
        x1 = tf.reshape(x1, [-1, 1])
        y1 = tf.reshape(y1, [-1, 1])
        h1 = tf.reshape(h1, [-1, 1])
        w1 = tf.reshape(w1, [-1, 1])
        theta1 = tf.reshape(theta1, [-1, 1])
        x2 = tf.reshape(x2, [-1, 1])
        y2 = tf.reshape(y2, [-1, 1])
        h2 = tf.reshape(h2, [-1, 1])
        w2 = tf.reshape(w2, [-1, 1])
        theta2 = tf.reshape(theta2, [-1, 1])
        theta1 *= np.pi / 180
        theta2 *= np.pi / 180

        # convert values to the probIoU
        aa = w1 ** 2 / 12
        bb = h1 ** 2 / 12
        angles = theta1
        a1 = aa * tf.math.pow(tf.math.cos(angles), 2.) + bb * tf.math.pow(tf.math.sin(angles), 2.)
        b1 = aa * tf.math.pow(tf.math.sin(angles), 2.) + bb * tf.math.pow(tf.math.cos(angles), 2.)
        c1 = 0.5 * (aa - bb) * tf.math.sin(2. * angles)

        aa = w2 ** 2 / 12
        bb = h2 ** 2 / 12
        angles = theta2
        a2 = aa * tf.math.pow(tf.math.cos(angles), 2.) + bb * tf.math.pow(tf.math.sin(angles), 2.)
        b2 = aa * tf.math.pow(tf.math.sin(angles), 2.) + bb * tf.math.pow(tf.math.cos(angles), 2.)
        c2 = 0.5 * (aa - bb) * tf.math.sin(2. * angles)

        # compute probIoU
        B1 = 1 / 4. * ((a1 + a2) * (y1 - y2) ** 2. + (b1 + b2) * (x1 - x2) ** 2.) + 1 / 2. * (
        (c1 + c2) * (x2 - x1) * (y1 - y2))
        B1 /= ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2. + 1e-3)

        sqrt = (a1 * b1 - c1 ** 2) * (a2 * b2 - c2 ** 2)
        sqrt = tf.clip_by_value(sqrt, 1e-3, tf.reduce_max(sqrt) + 1e-3)
        B2 = ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2.) / (4. * tf.math.sqrt(sqrt) + 1e-3)
        B2 = tf.clip_by_value(B2, 1e-3, tf.reduce_max(B2) + 1e-3)
        B2 = 1 / 2. * tf.math.log(B2)

        Bd = B1 + B2
        Bd = tf.clip_by_value(Bd, 1e-3, 100.)

        l1 = tf.math.sqrt(1 - tf.math.exp(-Bd) + 1e-3)

        if mode == 'l3':
            l2 = tf.math.pow(l1, 2.)
            probiou = - tf.math.log(1. - l2 + 1e-3)
        else:
            probiou = l1

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(probiou) / normalizer
