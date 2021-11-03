# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def clip_boxes_to_img_boundaries(decode_boxes, img_shape):
    '''
    :param decode_boxes:
    :return: decode boxes, and already clip to boundaries
    '''

    with tf.name_scope('clip_boxes_to_img_boundaries'):

        # xmin, ymin, xmax, ymax = tf.unstack(decode_boxes, axis=1)
        xmin = decode_boxes[:, 0]
        ymin = decode_boxes[:, 1]
        xmax = decode_boxes[:, 2]
        ymax = decode_boxes[:, 3]
        img_h, img_w = img_shape[1], img_shape[2]

        img_h, img_w = tf.cast(img_h, tf.float32), tf.cast(img_w, tf.float32)

        xmin = tf.maximum(tf.minimum(xmin, img_w-1.), 0.)
        ymin = tf.maximum(tf.minimum(ymin, img_h-1.), 0.)

        xmax = tf.maximum(tf.minimum(xmax, img_w-1.), 0.)
        ymax = tf.maximum(tf.minimum(ymax, img_h-1.), 0.)

        return tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))


def filter_outside_boxes(boxes, img_h, img_w):
    '''
    :param anchors:boxes with format [xmin, ymin, xmax, ymax]
    :param img_h: height of image
    :param img_w: width of image
    :return: indices of anchors that inside the image boundary
    '''

    with tf.name_scope('filter_outside_boxes'):
        xmin, ymin, xmax, ymax = tf.unstack(boxes, axis=1)

        xmin_index = tf.greater_equal(xmin, 0)
        ymin_index = tf.greater_equal(ymin, 0)
        xmax_index = tf.less_equal(xmax, tf.cast(img_w, tf.float32))
        ymax_index = tf.less_equal(ymax, tf.cast(img_h, tf.float32))

        indices = tf.transpose(tf.stack([xmin_index, ymin_index, xmax_index, ymax_index]))
        indices = tf.cast(indices, dtype=tf.int32)
        indices = tf.reduce_sum(indices, axis=1)
        indices = tf.where(tf.equal(indices, 4))
        # indices = tf.equal(indices, 4)
        return tf.reshape(indices, [-1])


def padd_boxes_with_zeros(boxes, scores, max_num_of_boxes):

    '''
    num of boxes less than max num of boxes, so it need to pad with zeros[0, 0, 0, 0]
    :param boxes:
    :param scores: [-1]
    :param max_num_of_boxes:
    :return:
    '''

    pad_num = tf.cast(max_num_of_boxes, tf.int32) - tf.shape(boxes)[0]

    zero_boxes = tf.zeros(shape=[pad_num, 4], dtype=boxes.dtype)
    zero_scores = tf.zeros(shape=[pad_num], dtype=scores.dtype)

    final_boxes = tf.concat([boxes, zero_boxes], axis=0)

    final_scores = tf.concat([scores, zero_scores], axis=0)

    return final_boxes, final_scores