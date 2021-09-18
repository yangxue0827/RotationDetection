# -*- coding: utf-8 -*-
# Author: Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>
#
# License: Apache-2.0 license

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from libs.models.losses.losses_gwd import LossGWD
from libs.utils import bbox_transform
from libs.utils.coordinate_convert import coordinate_present_convert
from utils.quad2rbox import quad2rbox_tf
from utils.order_points import re_order


class LossKL(LossGWD):

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

    def KL_divergence(self, mu1, mu2, mu1_T, mu2_T, sigma1, sigma2):
        sigma1_square = tf.linalg.matmul(sigma1, sigma1)
        sigma2_square = tf.linalg.matmul(sigma2, sigma2)
        item1 = tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(sigma2_square), sigma1_square))
        item2 = tf.linalg.matmul(tf.linalg.matmul(mu2-mu1, tf.linalg.inv(sigma2_square)), mu2_T-mu1_T)
        item3 = tf.log(tf.linalg.det(sigma2_square) / (tf.linalg.det(sigma1_square) + 1e-4))
        return (item1 + item2 + item3 - 2) / 2.

    def KL_divergence_(self, mu1, mu2, mu1_T, mu2_T, sigma1, sigma2):
        """
        Need large weight
        """
        sigma1_square = tf.linalg.matmul(sigma1, sigma1)
        sigma2_square = tf.linalg.matmul(sigma2, sigma2)
        item1 = tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(sigma2_square), sigma1_square))
        item2 = tf.linalg.matmul(tf.linalg.matmul(mu2-mu1, tf.linalg.inv(sigma2_square)), mu2_T-mu1_T)
        item3 = tf.log(tf.linalg.det(sigma2_square) / (tf.linalg.det(sigma1_square) + 1e-4))
        item1 = tf.reshape(item1, [-1, ])
        item2 = tf.reshape(item2, [-1, ])
        item3 = tf.reshape(item3, [-1, ])
        return (item1 + item2 + item3 - 2) / 2.

    def KL_divergence_loss(self, preds, anchor_state, target_boxes, anchors, is_refine=False, tau=1.0, func=0, shrink_ratio=1.):
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

        # ratio = tf.maximum(preds[:, 2] / preds[:, 3], preds[:, 3] / preds[:, 2])
        #
        # indices = tf.reshape(tf.where(tf.less(ratio, 500)), [-1, ])
        # preds = tf.gather(preds, indices)
        # target_boxes = tf.gather(target_boxes, indices)
        # anchors = tf.gather(anchors, indices)

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

        # sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(boxes_pred, target_boxes_)  # D(Np||Nt)
        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(target_boxes_, boxes_pred, shrink_ratio=shrink_ratio)  # D(Nt||Np)

        # KL_divergence need normalizer == KL_divergence_ do not need normalizer
        KL_distance = tf.reshape(self.KL_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        # KL_distance = tf.reshape(self.KL_divergence_(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        KL_distance = tf.maximum(KL_distance, 0.0)

        if func == 0:
            KL_distance = tf.maximum(tf.sqrt(KL_distance), 0.)
        elif func == 1:
            KL_distance = tf.maximum(tf.log(KL_distance + 1.), 0.)
        else:
            KL_distance = tf.maximum(tf.log(tf.sqrt(KL_distance) + 1.), 0.)

        if True:
            KL_similarity = 1 / (KL_distance + tau)
            KL_loss = (1 - KL_similarity) * 0.05
        else:
            KL_loss = KL_distance * 0.05

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(KL_loss) / normalizer

    def KL_divergence_max_loss(self, preds, anchor_state, target_boxes, anchors,
                               is_refine=False, tau=1.0, func=0):
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

        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(boxes_pred, target_boxes_)

        KL_distance1 = tf.reshape(self.KL_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        KL_distance1 = tf.maximum(KL_distance1, 0.0)

        KL_distance2 = tf.reshape(self.KL_divergence(mu2, mu1, mu2_T, mu1_T, sigma2, sigma1), [-1, 1])
        KL_distance2 = tf.maximum(KL_distance2, 0.0)

        KL_distance = tf.maximum(KL_distance1, KL_distance2)

        if func == 0:
            KL_distance = tf.maximum(tf.sqrt(KL_distance), 0.)
        else:
            KL_distance = tf.maximum(tf.log(KL_distance + 1.), 0.)

        if True:
            KL_similarity = 1 / (KL_distance + tau)
            KL_loss = (1 - KL_similarity) * 0.05
        else:
            KL_loss = KL_distance * 0.05

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(KL_loss) / normalizer

    def KL_divergence_min_loss(self, preds, anchor_state, target_boxes, anchors,
                               is_refine=False, tau=1.0, func=0):
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

        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(boxes_pred, target_boxes_)

        KL_distance1 = tf.reshape(self.KL_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        KL_distance1 = tf.maximum(KL_distance1, 0.0)

        KL_distance2 = tf.reshape(self.KL_divergence(mu2, mu1, mu2_T, mu1_T, sigma2, sigma1), [-1, 1])
        KL_distance2 = tf.maximum(KL_distance2, 0.0)

        KL_distance = tf.minimum(KL_distance1, KL_distance2)

        if func == 0:
            KL_distance = tf.maximum(tf.sqrt(KL_distance), 0.)
        else:
            KL_distance = tf.maximum(tf.log(KL_distance + 1.), 0.)

        if True:
            KL_similarity = 1 / (KL_distance + tau)
            KL_loss = (1 - KL_similarity) * 0.05
        else:
            KL_loss = KL_distance * 0.05

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(KL_loss) / normalizer

    def jeffreys_divergence_loss(self, preds, anchor_state, target_boxes, anchors, is_refine=False, tau=1.0, func=0):
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

        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(boxes_pred, target_boxes_)

        KL_distance1 = tf.reshape(self.KL_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        KL_distance1 = tf.maximum(KL_distance1, 0.0)

        KL_distance2 = tf.reshape(self.KL_divergence(mu2, mu1, mu2_T, mu1_T, sigma2, sigma1), [-1, 1])
        KL_distance2 = tf.maximum(KL_distance2, 0.0)

        KL_distance = KL_distance1 + KL_distance2

        if func == 0:
            KL_distance = tf.maximum(tf.sqrt(KL_distance), 0.)
        else:
            KL_distance = tf.maximum(tf.log(KL_distance + 1.), 0.)

        if True:
            KL_similarity = 1 / (KL_distance + tau)
            KL_loss = (1 - KL_similarity) * 0.05
        else:
            KL_loss = KL_distance * 0.05

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(KL_loss) / normalizer

    def JS_divergence(self, mu1, mu2, mu1_T, mu2_T, sigma1, sigma2):

        def kl_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2):
            item1 = tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(sigma2), sigma1))
            item2 = tf.linalg.matmul(tf.linalg.matmul(mu2 - mu1, tf.linalg.inv(sigma2)), mu2_T - mu1_T)
            item3 = tf.log(tf.linalg.det(sigma2) / (tf.linalg.det(sigma1) + 1e-4))
            return (item1 + item2 + item3 - 2) / 2.

        sigma1_square = tf.linalg.matmul(sigma1, sigma1)
        sigma2_square = tf.linalg.matmul(sigma2, sigma2)

        sigma_square = (sigma1_square + sigma2_square) / 2.
        mu = (mu1 + mu2) / 2.
        mu_T = (mu1_T + mu2_T) / 2.

        js1 = kl_divergence(mu1, mu, mu1_T, mu_T, sigma1_square, sigma_square)
        js2 = kl_divergence(mu2, mu, mu2_T, mu_T, sigma2_square, sigma_square)
        js = js1 + js2

        return js

    def JS_divergence_loss(self, preds, anchor_state, target_boxes, anchors,
                           is_refine=False, tau=1.0, func=0):
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

        ratio = tf.maximum(preds[:, 2] / preds[:, 3], preds[:, 3] / preds[:, 2])

        indices = tf.reshape(tf.where(tf.less(ratio, 150)), [-1, ])
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

        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(boxes_pred, target_boxes_)

        # sigma1, sigma2 = tf.py_func(self.debug,
        #                             inp=[mu1, mu2, mu1_T, mu2_T, tf.linalg.matmul(sigma1, sigma1),
        #                                  tf.linalg.matmul(sigma2, sigma2)],
        #                             Tout=[tf.float32, tf.float32])
        # sigma1 = tf.reshape(sigma1, [-1, 2, 2])
        # sigma2 = tf.reshape(sigma2, [-1, 2, 2])

        JS_distance = tf.reshape(self.JS_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        JS_distance = tf.maximum(JS_distance, 0.0)

        if func == 0:
            JS_distance = tf.maximum(tf.sqrt(JS_distance), 0.)
        else:
            JS_distance = tf.maximum(tf.log(JS_distance + 1.), 0.)

        if True:
            JS_similarity = 1 / (JS_distance + tau)
            JS_loss = (1 - JS_similarity) * 0.05
        else:
            JS_loss = JS_distance * 0.05

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(JS_loss) / normalizer

    def KL_divergence_quad_loss(self, preds, anchor_state, target_boxes, anchors, is_refine=False, tau=1.0, func=0):

        target_boxes = tf.reshape(target_boxes[:, :-1], [-1, 8])

        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
        preds = tf.gather(preds, indices)
        target_boxes = tf.gather(target_boxes, indices)
        anchors = tf.gather(anchors, indices)

        if self.cfgs.METHOD == 'H' and not is_refine:
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            w = anchors[:, 2] - anchors[:, 0] + 1
            h = anchors[:, 3] - anchors[:, 1] + 1
            # theta = -90 * tf.ones_like(x_c)
            anchors = tf.transpose(tf.stack([x_c, y_c, w, h]))

        boxes_pred = bbox_transform.qbbox_transform_inv(boxes=anchors, deltas=preds)
        target_boxes_ = tf.py_func(func=re_order,
                                   inp=[target_boxes],
                                   Tout=[tf.float32])
        target_boxes_ = tf.reshape(target_boxes_, [-1, 8])
        x1, y1, x2, y2, x3, y3, x4, y4 = tf.unstack(target_boxes_, axis=-1)
        boxes_pred = quad2rbox_tf(boxes_pred)

        target_boxes_1 = quad2rbox_tf(tf.transpose(tf.stack([x1, y1, x2, y2, x3, y3, x4, y4])))
        sigma1_1, sigma2_1, mu1_1, mu2_1, mu1_T_1, mu2_T_1 = self.get_gaussian_param(boxes_pred, target_boxes_1)
        KL_distance_1 = tf.reshape(self.KL_divergence(mu1_1, mu2_1, mu1_T_1, mu2_T_1, sigma1_1, sigma2_1), [-1, 1])
        KL_distance_1 = tf.maximum(KL_distance_1, 0.0)

        target_boxes_2 = quad2rbox_tf(tf.transpose(tf.stack([x2, y2, x3, y3, x4, y4, x1, y1])))
        sigma1_2, sigma2_2, mu1_2, mu2_2, mu1_T_2, mu2_T_2 = self.get_gaussian_param(boxes_pred, target_boxes_2)
        KL_distance_2 = tf.reshape(self.KL_divergence(mu1_2, mu2_2, mu1_T_2, mu2_T_2, sigma1_2, sigma2_2), [-1, 1])
        KL_distance_2 = tf.maximum(KL_distance_2, 0.0)

        target_boxes_3 = quad2rbox_tf(tf.transpose(tf.stack([x4, y4, x1, y1, x2, y2, x3, y3])))
        sigma1_3, sigma2_3, mu1_3, mu2_3, mu1_T_3, mu2_T_3 = self.get_gaussian_param(boxes_pred, target_boxes_3)
        KL_distance_3 = tf.reshape(self.KL_divergence(mu1_3, mu2_3, mu1_T_3, mu2_T_3, sigma1_3, sigma2_3), [-1, 1])
        KL_distance_3 = tf.maximum(KL_distance_3, 0.0)

        KL_distance = tf.minimum(tf.minimum(KL_distance_1, KL_distance_2), KL_distance_3)

        if func == 0:
            KL_distance = tf.maximum(tf.sqrt(KL_distance), 0.)
        else:
            KL_distance = tf.maximum(tf.log(KL_distance + 1.), 0.)

        if True:
            KL_similarity = 1 / (KL_distance + tau)
            KL_loss = (1 - KL_similarity) * 0.05
        else:
            KL_loss = KL_distance * 0.05

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(KL_loss) / normalizer

    def KL_divergence_loss_two_stage_(self, bbox_pred, rois, target_gt, label, num_classes, tau=1.0, func=0):
        '''
        :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 5]
        :param label:[-1]
        :param num_classes:
        :param sigma:
        :return:
        '''

        outside_mask = tf.reshape(tf.stop_gradient(tf.to_float(tf.greater(label, 0))), [-1, 1])

        target_gt = tf.reshape(tf.tile(tf.reshape(target_gt, [-1, 1, 5]), [1, num_classes, 1]), [-1, 5])
        x_c = (rois[:, 2] + rois[:, 0]) / 2
        y_c = (rois[:, 3] + rois[:, 1]) / 2
        h = rois[:, 2] - rois[:, 0] + 1
        w = rois[:, 3] - rois[:, 1] + 1
        theta = -90 * tf.ones_like(x_c)
        rois = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
        rois = tf.reshape(tf.tile(tf.reshape(rois, [-1, 1, 5]), [1, num_classes, 1]), [-1, 5])

        boxes_pred = bbox_transform.rbbox_transform_inv(boxes=rois, deltas=tf.reshape(bbox_pred, [-1, 5]),
                                                        scale_factors=self.cfgs.ROI_SCALE_FACTORS)

        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(boxes_pred, target_gt)

        KL_distance = tf.reshape(self.KL_divergence_(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        KL_distance = tf.maximum(KL_distance, 0.0)

        if func == 0:
            KL_distance = tf.maximum(tf.sqrt(KL_distance), 0.)
        else:
            KL_distance = tf.maximum(tf.log(KL_distance + 1.), 0.)

        if True:
            KL_similarity = 1 / (KL_distance + tau)
            KL_loss = (1 - KL_similarity) * 0.05
        else:
            KL_loss = KL_distance * 0.05

        KL_loss = tf.reshape(KL_loss, [-1, num_classes])

        inside_mask = tf.one_hot(tf.reshape(label, [-1, ]), depth=num_classes, axis=-1)
        inside_mask = tf.stop_gradient(tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

        normalizer = tf.stop_gradient(tf.where(tf.greater_equal(label, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        # normalizer = tf.to_float(tf.shape(inside_mask)[0])
        normalizer = tf.maximum(1.0, normalizer)
        bbox_loss = tf.reduce_sum(
            tf.reduce_sum(KL_loss * inside_mask, 1) * outside_mask) / normalizer

        return bbox_loss

    def KL_divergence_loss_two_stage(self, bbox_pred, rois, target_gt, label, num_classes, tau=1.0, func=0):

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

        KL_distance = tf.reshape(self.KL_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        KL_distance = tf.maximum(KL_distance, 0.0)

        if func == 0:
            KL_distance = tf.maximum(tf.sqrt(KL_distance), 0.)
        elif func == 1:
            KL_distance = tf.maximum(tf.log(KL_distance + 1.), 0.)
        else:
            KL_distance = tf.maximum(tf.log(tf.sqrt(KL_distance) + 1.), 0.)

        if True:
            KL_similarity = 1 / (KL_distance + tau)
            KL_loss = (1 - KL_similarity) * 0.05
        else:
            KL_loss = KL_distance * 0.05

        normalizer = tf.stop_gradient(tf.where(tf.greater(label, 0)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(KL_loss) / normalizer

    def KL_divergence_pyramid_loss(self, preds, anchor_state, target_boxes, anchors, is_refine=False, tau=1.0, func=0, shrink_ratio=[]):
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

        # ratio = tf.maximum(preds[:, 2] / preds[:, 3], preds[:, 3] / preds[:, 2])
        #
        # indices = tf.reshape(tf.where(tf.less(ratio, 500)), [-1, ])
        # preds = tf.gather(preds, indices)
        # target_boxes = tf.gather(target_boxes, indices)
        # anchors = tf.gather(anchors, indices)

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

        # sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(boxes_pred, target_boxes_)  # D(Np||Nt)
        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(target_boxes_, boxes_pred)  # D(Nt||Np)

        # KL_divergence need normalizer == KL_divergence_ do not need normalizer
        KL_distance = tf.reshape(self.KL_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        # KL_distance = tf.reshape(self.KL_divergence_(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        KL_distance = tf.maximum(KL_distance, 0.0)

        if func == 0:
            KL_distance = tf.maximum(tf.sqrt(KL_distance), 0.)
        elif func == 1:
            KL_distance = tf.maximum(tf.log(KL_distance + 1.), 0.)
        else:
            KL_distance = tf.maximum(tf.log(tf.sqrt(KL_distance) + 1.), 0.)

        if True:
            KL_similarity = 1 / (KL_distance + tau)
            KL_loss = (1 - KL_similarity) * 0.05
        else:
            KL_loss = KL_distance * 0.05

        for sr in shrink_ratio:
            # sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(boxes_pred, target_boxes_)  # D(Np||Nt)
            sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(target_boxes_, boxes_pred, shrink_ratio=sr)  # D(Nt||Np)

            # KL_divergence need normalizer == KL_divergence_ do not need normalizer
            KL_distance = tf.reshape(self.KL_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
            # KL_distance = tf.reshape(self.KL_divergence_(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
            KL_distance = tf.maximum(KL_distance, 0.0)

            if func == 0:
                KL_distance = tf.maximum(tf.sqrt(KL_distance), 0.)
            elif func == 1:
                KL_distance = tf.maximum(tf.log(KL_distance + 1.), 0.)
            else:
                KL_distance = tf.maximum(tf.log(tf.sqrt(KL_distance) + 1.), 0.)

            if True:
                KL_similarity = 1 / (KL_distance + tau)
                KL_loss += (1 - KL_similarity) * 0.05
            else:
                KL_loss += KL_distance * 0.05

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(KL_loss) / (normalizer * (len(shrink_ratio) + 1))
