# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils.densely_coded_label import angle_label_decode
from libs.models.losses.losses import Loss


class LossCSL(Loss):

    def angle_focal_loss(self, labels, pred, anchor_state, alpha=0.25, gamma=2.0):

        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
        labels = tf.gather(labels, indices)
        pred = tf.gather(pred, indices)

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
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    per_entry_cross_ent)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
        # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

        return tf.reduce_sum(focal_cross_entropy_loss) / normalizer

    def angle_cls_focal_loss(self, labels, pred, anchor_state, alpha=None, gamma=2.0, decimal_weight=None):

        indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
        labels = tf.gather(labels, indices)
        pred = tf.gather(pred, indices)
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

        if decimal_weight is not None:
            angle_decode_labels = tf.py_func(func=angle_label_decode,
                                             inp=[labels, self.cfgs.ANGLE_RANGE, self.cfgs.OMEGA, self.cfgs.ANGLE_MODE],
                                             Tout=[tf.float32])
            angle_decode_labels = tf.reshape(angle_decode_labels, [-1, ]) * -1

            angle_decode_pred = tf.py_func(func=angle_label_decode,
                                           inp=[tf.sigmoid(pred), self.cfgs.ANGLE_RANGE, self.cfgs.OMEGA, self.cfgs.ANGLE_MODE],
                                           Tout=[tf.float32])

            angle_decode_pred = tf.reshape(angle_decode_pred, [-1, ]) * -1

            diff_weight = tf.reshape(tf.log(abs(angle_decode_labels - angle_decode_pred) + 1), [-1, 1])
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

