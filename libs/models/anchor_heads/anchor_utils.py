
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import tensorflow as tf
import random
import numpy as np

# def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
#                      scales=2 ** np.arange(3, 6)):
#   """
#   Generate anchor (reference) windows by enumerating aspect ratios X
#   scales wrt a reference (0, 0, 15, 15) window.
#   """
#
#   base_anchor = np.array([1, 1, base_size, base_size]) - 1
#   ratio_anchors = _ratio_enum(base_anchor, ratios)
#   anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
#                        for i in range(ratio_anchors.shape[0])])
#   return anchors
#
# def _whctrs(anchor):
#   """
#   Return width, height, x center, and y center for an anchor (window).
#   """
#
#   w = anchor[2] - anchor[0] + 1
#   h = anchor[3] - anchor[1] + 1
#   x_ctr = anchor[0] + 0.5 * (w - 1)
#   y_ctr = anchor[1] + 0.5 * (h - 1)
#   return w, h, x_ctr, y_ctr
#
#
# def _mkanchors(ws, hs, x_ctr, y_ctr):
#   """
#   Given a vector of widths (ws) and heights (hs) around a center
#   (x_ctr, y_ctr), output a set of anchors (windows).
#   """
#
#   ws = ws[:, np.newaxis]
#   hs = hs[:, np.newaxis]
#   anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
#                        y_ctr - 0.5 * (hs - 1),
#                        x_ctr + 0.5 * (ws - 1),
#                        y_ctr + 0.5 * (hs - 1)))
#   return anchors
#
#
# def _ratio_enum(anchor, ratios):
#   """
#   Enumerate a set of anchors for each aspect ratio wrt an anchor.
#   """
#
#   w, h, x_ctr, y_ctr = _whctrs(anchor)
#   size = w * h
#   size_ratios = size / ratios
#   ws = np.round(np.sqrt(size_ratios))
#   hs = np.round(ws * ratios)
#   anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
#   return anchors
#
#
# def _scale_enum(anchor, scales):
#   """
#   Enumerate a set of anchors for each scale wrt an anchor.
#   """
#
#   w, h, x_ctr, y_ctr = _whctrs(anchor)
#   ws = w * scales
#   hs = h * scales
#   anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
#   return anchors
#
#
# def make_anchors(
#         height, width, feat_stride, anchor_scales=(8, 16, 32),
#         anchor_ratios=(0.5, 1., 2.), base_size=16):
#
#     anchors = generate_anchors(
#         ratios=np.array(anchor_ratios), scales=np.array(anchor_scales),
#         base_size=base_size)
#     shift_x = tf.range(width, dtype=np.float32) * feat_stride
#     shift_y = tf.range(height, dtype=np.float32) * feat_stride
#     shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
#     shifts = tf.stack(
#         (tf.reshape(shift_x, (-1, 1)), tf.reshape(shift_y, (-1, 1)),
#          tf.reshape(shift_x, (-1, 1)), tf.reshape(shift_y, (-1, 1))))
#     shifts = tf.transpose(shifts, [1, 0, 2])
#     final_anc = tf.constant(anchors.reshape((1, -1, 4)), dtype=np.float32) + \
#           tf.transpose(tf.reshape(shifts, (1, -1, 4)), (1, 0, 2))
#     return tf.reshape(final_anc, (-1, 4))

#
def make_anchors(base_anchor_size, anchor_scales, anchor_ratios,
                 featuremap_height, featuremap_width,
                 stride, name='make_anchors'):
    '''
    :param base_anchor_size:256
    :param anchor_scales:
    :param anchor_ratios:
    :param featuremap_height:
    :param featuremap_width:
    :param stride:
    :return:
    '''
    with tf.variable_scope(name):
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)  # [x_center, y_center, w, h]

        ws, hs = enum_ratios(enum_scales(base_anchor, anchor_scales),
                             anchor_ratios)  # per locations ws and hs

        x_centers = tf.range(featuremap_width, dtype=tf.float32) * stride
        y_centers = tf.range(featuremap_height, dtype=tf.float32) * stride

        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)

        anchor_centers = tf.stack([x_centers, y_centers], 2)
        anchor_centers = tf.reshape(anchor_centers, [-1, 2])

        box_sizes = tf.stack([ws, hs], axis=2)
        box_sizes = tf.reshape(box_sizes, [-1, 2])
        # anchors = tf.concat([anchor_centers, box_sizes], axis=1)
        anchors = tf.concat([anchor_centers - 0.5*box_sizes,
                             anchor_centers + 0.5*box_sizes], axis=1)
        return anchors


def enum_scales(base_anchor, anchor_scales):

    anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))

    return anchor_scales


def enum_ratios(anchors, anchor_ratios):
    '''
    ratio = h /w
    :param anchors:
    :param anchor_ratios:
    :return:
    '''
    ws = anchors[:, 2]  # for base anchor: w == h
    hs = anchors[:, 3]
    sqrt_ratios = tf.sqrt(tf.constant(anchor_ratios))

    ws = tf.reshape(ws / sqrt_ratios[:, tf.newaxis], [-1, 1])
    hs = tf.reshape(hs * sqrt_ratios[:, tf.newaxis], [-1, 1])

    return ws, hs


def shift_anchor(anchors, stride):
    shift_delta = [(stride // 2, 0, stride // 2, 0),
                   (0, stride // 2, 0, stride // 2),
                   (stride // 2, stride // 2, stride // 2, stride // 2)]
    coord = tf.unstack(anchors, axis=1)
    anchors_shift = [anchors]
    for delta in shift_delta:
        coord_shift = []
        for sd, c in zip(delta, coord):
            coord_shift.append(sd + c)
        tmp = tf.transpose(tf.stack(coord_shift))
        anchors_shift.append(tmp)
    anchors_ = tf.concat(anchors_shift, axis=1)
    anchors_ = tf.reshape(anchors_, [-1, 4])

    return anchors_


def shift_jitter(anchors, stride):
    pass


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    base_anchor_size = 16
    anchor_scales = [1.]
    anchor_ratios = [1.0]
    anchors = make_anchors(base_anchor_size=base_anchor_size, anchor_ratios=anchor_ratios,
                           anchor_scales=anchor_scales,
                           featuremap_width=5,
                           featuremap_height=5,
                           stride=16)
    anchors_ = shift_anchor(anchors, 16)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        anchor_result = sess.run(anchors_)
        print(anchor_result)