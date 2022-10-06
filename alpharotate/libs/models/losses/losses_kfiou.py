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


class LossKF(Loss):

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

    def kalman_filter(self, mu1, mu2, sigma1, sigma2):
        sigma1_square = tf.linalg.matmul(sigma1, sigma1)
        sigma2_square = tf.linalg.matmul(sigma2, sigma2)
        k = tf.linalg.matmul(sigma1_square, tf.linalg.inv(sigma1_square + sigma2_square))
        mu = mu1 + tf.linalg.matmul(k, mu2 - mu1)
        sigma_square = sigma1_square - tf.linalg.matmul(k, sigma1_square)
        return mu, sigma_square

    def kalman_filter2(self, mu1, mu2, sigma1, sigma2):
        sigma1_square = tf.linalg.matmul(sigma1, sigma1)
        sigma2_square = tf.linalg.matmul(sigma2, sigma2)
        k = tf.linalg.inv(tf.linalg.inv(sigma1_square) + tf.linalg.inv(sigma2_square))
        mu = tf.linalg.matmul(tf.linalg.inv(sigma1_square), mu1) + tf.linalg.matmul(tf.linalg.inv(sigma2_square), mu2)
        mu = tf.linalg.matmul(k, mu)
        return mu, k

    def kalman_filter_iou(self, preds, anchor_state, target_boxes, anchors, is_refine=False, mode=0):
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

        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(target_boxes_, boxes_pred)

        # yx_loss = tf.linalg.matmul(tf.linalg.matmul(mu2 - mu1, tf.linalg.inv(tf.linalg.matmul(sigma2, sigma2))), mu2_T - mu1_T)

        # 计算两个矩形的协方差
        # sigma1 = tf.linalg.matmul(sigma1, sigma1)
        # sigma2 = tf.linalg.matmul(sigma2, sigma2)
        # 求卡尔曼滤波的结果
        # *************************************************************************
        # sigma1_inv, sigma2_inv = tf.linalg.inv(sigma1), tf.linalg.inv(sigma2)
        # n1 = tf.linalg.matmul(sigma1_inv, mu1)
        # n2 = tf.linalg.matmul(sigma2_inv, mu2)
        # sigma_inv = sigma1_inv + sigma2_inv
        # n = n1 + n2
        # # scaler = tf.linalg.matmul(mu1-mu2, tf.linalg.matrix_transpose(mu1-mu2))
        # sigma = tf.linalg.inv(sigma_inv)
        # additional_term = tf.exp((-2 * tf.log(2 * np.pi) - tf.log(tf.linalg.det(sigma_inv)) + tf.squeeze(
        #     tf.linalg.matmul(tf.linalg.matmul(tf.matrix_transpose(n), sigma), n)) + tf.log(
        #     tf.linalg.det(sigma1_inv)) + tf.log(tf.linalg.det(sigma2_inv)) - tf.squeeze(
        #     tf.linalg.matmul(tf.linalg.matmul(tf.matrix_transpose(n1), sigma1), n1)) - tf.squeeze(
        #     tf.linalg.matmul(tf.linalg.matmul(tf.matrix_transpose(n2), sigma2),
        #                      n2))) / 2.0)  # additional term on sigma
        # # additional_term = tf.squeeze(additional_term)
        #
        # # additional_term = tf.reshape(1 / tf.sqrt(2*np.pi*(tf.linalg.det(sigma1)+tf.linalg.det(sigma2))), shape=[-1,1,1]) * tf.exp(-scaler / tf.reshape(2*(tf.linalg.det(sigma1)+tf.linalg.det(sigma2)), shape=[-1,1,1]))
        # # additional_term = tf.reshape(tf.sqrt(2*np.pi*(tf.linalg.det(sigma))), shape=[-1,1,1]) * tf.exp(-scaler * tf.reshape(2*(tf.linalg.det(sigma)), shape=[-1,1,1]))
        # # additional_term = scaler
        # # additional_term = tf.sqrt(tf.sqrt(tf.sqrt(tf.sqrt(additional_term))))
        # info = additional_term
        #
        # mu = tf.linalg.matmul(sigma, n)
        # scale = 100000
        # sigma = sigma * tf.reshape(additional_term, shape=[-1, 1, 1]) * scale
        # c1 = 2 * tf.log(2 * math.pi)
        # c2 = tf.log(tf.linalg.det(sigma1_inv))
        # c3 = tf.linalg.matmul(tf.linalg.matmul(tf.linalg.transpose(n1), tf.linalg.inv(sigma1_inv)), n1)
        # # c = -0.5 * (2 * tf.log(2 * math.pi) - tf.log(tf.linalg.det(sigma1_inv))
        # #             + tf.linalg.matmul(tf.linalg.matmul(tf.linalg.transpose(n1), tf.linalg.inv(sigma1_inv)), n1))
        # c = tf.exp(-0.5 * (c1 - c2 + tf.reshape(c3, [-1, ])))
        mu, sigma = self.kalman_filter(mu1_T, mu2_T, sigma1, sigma2)
        # *************************************************************************
        eig = tf.reshape(tf.linalg.eigvalsh(sigma), [-1, 2])  # 根据卡尔曼滤波的协方差求特征值，也就是w^2/4, h^2/4
        if mode == 0:
            overlap = tf.sqrt(eig[:, 0] * eig[:, 1]) * 4  # * additional_term**2 # 高斯反求矩形的wh，计算面积
        else:
            det, l = eig[:, 0] * eig[:, 1], 1e-5
            tmp = tf.linalg.matmul(sigma1, sigma1) + tf.linalg.matmul(sigma2, sigma2)
            item1 = -2 * math.pi * tf.sqrt(det) * tf.log(2 * math.pi * tf.sqrt(tf.linalg.det(sigma)) * l)
            item2 = -2 * math.pi * tf.sqrt(det) * tf.log(1 / (
                tf.sqrt(tf.linalg.det(2 * math.pi * tmp))))
            item3 = tf.cast(tf.linalg.matmul(tf.linalg.matmul((mu1 + mu2), tf.linalg.inv(tmp)), (mu1_T + mu2_T)),
                            tf.float32)
            item3 = tf.reshape(item3, [-1, ]) * (-math.pi * tf.sqrt(det))

            overlap = item1 + item2 + item3
        aera1 = boxes_pred[:, 2] * boxes_pred[:, 3]
        aera2 = target_boxes_[:, 2] * target_boxes_[:, 3]
        # info = 1 / additional_term

        # c_1 = 1 / (2*math.pi*tf.sqrt(tf.linalg.det(sigma1)))
        # c_2 = 1 / (2*math.pi*tf.sqrt(tf.linalg.det(sigma2)))
        # c_3 = 1 / (2*math.pi*tf.sqrt(tf.linalg.det(sigma)))

        # c_2 = tf.linalg.matmul(tf.linalg.matmul(tf.linalg.transpose(mu - mu1), tf.linalg.inv(sigma1)), mu - mu1)
        # c_2 = tf.reshape(tf.exp(-0.5*c_2), [-1, ])
        # overlap *= (c/c_3)
        iou = overlap / (aera1 + aera2 - overlap)

        iou_loss = -1 * tf.log(iou + 1e-5)
        # iou_loss = 1 - iou
        # iou_loss = tf.exp(iou_loss) - 1

        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(iou_loss) / normalizer

    def kalman_filter_iou_xy(self, target_delta, preds, anchor_state, target_boxes, anchors, sigma=3.0, is_refine=False):

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
        target_delta = tf.gather(target_delta, indices)

        ######################################################################

        sigma_squared = sigma ** 2

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = preds[:, :2] - target_delta[:, :2]
        regression_diff = tf.abs(regression_diff)

        regression_loss = tf.where(
            tf.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        ######################################################################

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

        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(target_boxes_, boxes_pred)

        # sigma2_square = tf.linalg.matmul(sigma2, sigma2)
        # regression_loss = tf.linalg.matmul(tf.linalg.matmul(mu2 - mu1, tf.linalg.inv(sigma2_square)), mu2_T - mu1_T)

        mu, sigma = self.kalman_filter(mu1_T, mu2_T, sigma1, sigma2)

        eig = tf.reshape(tf.linalg.eigvalsh(sigma), [-1, 2])  # 根据卡尔曼滤波的协方差求特征值，也就是w^2/4, h^2/4
        overlap = tf.sqrt(eig[:, 0] * eig[:, 1]) * 4  # 高斯反求矩形的wh，计算面积

        aera1 = boxes_pred[:, 2] * boxes_pred[:, 3]
        aera2 = target_boxes_[:, 2] * target_boxes_[:, 3]

        iou = overlap / (aera1 + aera2 - overlap)

        # iou_loss = -1*tf.log(iou+1e-5)
        # iou_loss = 1 - 3*iou
        iou_loss = 1 - iou
        iou_loss = tf.exp(iou_loss) - 1

        ######################################################################
        # equivalence
        if True:
            iou_loss = tf.reshape(iou_loss, [-1, 1])
            regression_loss = tf.reshape(tf.reduce_sum(regression_loss, axis=-1), [-1, ])
            total_loss = tf.reshape(iou_loss + regression_loss, [-1, 1])
            normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
            normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
            normalizer = tf.maximum(1.0, normalizer)

            return tf.reduce_sum(total_loss) / normalizer
        else:
            total_loss = tf.reshape(iou_loss, [-1, 1]) + tf.reshape(tf.reduce_sum(regression_loss, axis=-1), [-1, 1])
            return tf.reduce_sum(total_loss)

    def kalman_filter_iou_kl(self, preds, anchor_state, target_boxes, anchors, is_refine=False):

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

        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(target_boxes_, boxes_pred)

        # center loss in KLD
        sigma1_square = tf.linalg.matmul(sigma1, sigma1)
        center_loss = tf.linalg.matmul(tf.linalg.matmul(mu2 - mu1, tf.linalg.inv(sigma1_square)), mu2_T - mu1_T)
        # center_loss = tf.sqrt(tf.maximum(center_loss, 0.))
        center_loss = tf.log(tf.maximum(center_loss, 0.) + 1.0)

        mu, sigma = self.kalman_filter(mu1_T, mu2_T, sigma1, sigma2)

        eig = tf.reshape(tf.linalg.eigvalsh(sigma), [-1, 2])  # 根据卡尔曼滤波的协方差求特征值，也就是w^2/4, h^2/4
        overlap = tf.sqrt(eig[:, 0] * eig[:, 1]) * 4  # 高斯反求矩形的wh，计算面积

        aera1 = boxes_pred[:, 2] * boxes_pred[:, 3]
        aera2 = target_boxes_[:, 2] * target_boxes_[:, 3]

        iou = overlap / (aera1 + aera2 - overlap)

        # iou_loss = -1*tf.log(iou+1e-5)
        # iou_loss = 1 - 3*iou
        iou_loss = 1 - iou
        iou_loss = tf.exp(iou_loss) - 1

        ######################################################################
        # equivalence
        if True:
            iou_loss = tf.reshape(iou_loss, [-1, 1])
            regression_loss = tf.reshape(tf.reduce_sum(center_loss, axis=-1), [-1, ])
            total_loss = tf.reshape(iou_loss + regression_loss, [-1, 1])
            normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
            normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
            normalizer = tf.maximum(1.0, normalizer)

            return tf.reduce_sum(total_loss) / normalizer
        else:
            total_loss = tf.reshape(iou_loss, [-1, 1]) + tf.reshape(tf.reduce_sum(center_loss, axis=-1), [-1, 1])
            return tf.reduce_sum(total_loss)

    def kalman_filter_iou_two_stage(self, bbox_pred, rois, target_gt, label, num_classes):

        one_hot = tf.one_hot(tf.reshape(label, [-1, ]), depth=num_classes, axis=-1)

        indices = tf.reshape(tf.where(tf.greater(label, 0)), [-1, ])
        bbox_pred = tf.gather(tf.reshape(bbox_pred, [-1, num_classes * 5]), indices)
        rois = tf.gather(tf.reshape(rois, [-1, 4]), indices)
        target_gt = tf.gather(tf.reshape(target_gt, [-1, 5]), indices)
        one_hot = tf.gather(tf.reshape(one_hot, [-1, num_classes]), indices)

        bbox_pred = tf.reshape(tf.cast(one_hot, tf.float32), [-1, num_classes, 1]) * tf.reshape(bbox_pred,
                                                                                                [-1, num_classes,
                                                                                                 5])
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

        mu, sigma = self.kalman_filter(mu1_T, mu2_T, sigma1, sigma2)
        eig = tf.reshape(tf.linalg.eigvalsh(sigma), [-1, 2])  # 根据卡尔曼滤波的协方差求特征值，也就是w^2/4, h^2/4
        overlap = tf.sqrt(eig[:, 0] * eig[:, 1]) * 4  # * additional_term**2 # 高斯反求矩形的wh，计算面积

        aera1 = boxes_pred[:, 2] * boxes_pred[:, 3]
        aera2 = target_gt[:, 2] * target_gt[:, 3]

        iou = overlap / (aera1 + aera2 - overlap)

        iou_loss = 1 - iou
        iou_loss = tf.exp(iou_loss) - 1

        normalizer = tf.stop_gradient(tf.where(tf.greater(label, 0)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(iou_loss) / normalizer

    def kalman_filter_iou_yx_two_stage(self, bbox_pred, bbox_targets, rois, target_gt, label, num_classes, sigma=1.0):

        one_hot = tf.one_hot(tf.reshape(label, [-1, ]), depth=num_classes, axis=-1)

        indices = tf.reshape(tf.where(tf.greater(label, 0)), [-1, ])
        bbox_pred = tf.gather(tf.reshape(bbox_pred, [-1, num_classes * 5]), indices)
        bbox_targets = tf.gather(tf.reshape(bbox_targets, [-1, num_classes * 5]), indices)
        rois = tf.gather(tf.reshape(rois, [-1, 4]), indices)
        target_gt = tf.gather(tf.reshape(target_gt, [-1, 5]), indices)
        one_hot = tf.gather(tf.reshape(one_hot, [-1, num_classes]), indices)

        bbox_pred = tf.reshape(tf.cast(one_hot, tf.float32), [-1, num_classes, 1]) * tf.reshape(bbox_pred, [-1, num_classes, 5])
        bbox_pred = tf.reduce_sum(bbox_pred, axis=1)
        bbox_targets = tf.reshape(tf.cast(one_hot, tf.float32), [-1, num_classes, 1]) * tf.reshape(bbox_targets, [-1, num_classes, 5])
        bbox_targets = tf.reduce_sum(bbox_targets, axis=1)

        sigma_squared = sigma ** 2

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = bbox_pred[:, :2] - bbox_targets[:, :2]
        regression_diff = tf.abs(regression_diff)

        regression_loss = tf.where(
            tf.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        x_c = (rois[:, 2] + rois[:, 0]) / 2
        y_c = (rois[:, 3] + rois[:, 1]) / 2
        h = rois[:, 2] - rois[:, 0] + 1
        w = rois[:, 3] - rois[:, 1] + 1
        theta = -90 * tf.ones_like(x_c)
        rois = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

        boxes_pred = bbox_transform.rbbox_transform_inv(boxes=rois, deltas=tf.reshape(bbox_pred, [-1, 5]),
                                                        scale_factors=self.cfgs.ROI_SCALE_FACTORS)

        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.get_gaussian_param(boxes_pred, target_gt)

        mu, sigma = self.kalman_filter(mu1_T, mu2_T, sigma1, sigma2)
        eig = tf.reshape(tf.linalg.eigvalsh(sigma), [-1, 2])  # 根据卡尔曼滤波的协方差求特征值，也就是w^2/4, h^2/4
        overlap = tf.sqrt(eig[:, 0] * eig[:, 1]) * 4  # * additional_term**2 # 高斯反求矩形的wh，计算面积

        aera1 = boxes_pred[:, 2] * boxes_pred[:, 3]
        aera2 = target_gt[:, 2] * target_gt[:, 3]

        iou = overlap / (aera1 + aera2 - overlap)

        iou_loss = 1 - iou
        iou_loss = tf.exp(iou_loss) - 1

        iou_loss = tf.reshape(iou_loss, [-1, 1])
        regression_loss = tf.reshape(tf.reduce_sum(regression_loss, axis=-1), [-1, ])
        total_loss = tf.reshape(iou_loss + regression_loss, [-1, 1])

        normalizer = tf.stop_gradient(tf.where(tf.greater(label, 0)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(total_loss) / normalizer




