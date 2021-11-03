# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def gereate_centering_anchor(
        base_size=16, ratios=[0.5, 1, 2],
        scales=2 ** np.arange(3, 6)):

    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    base_anchor = tf.convert_to_tensor(np.array([1, 1, base_size, base_size]) - (base_size // 2))
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = tf.cast(tf.stack([_scale_enum(ratio_anchors[i, :], scales)
                                for i in range(ratio_anchors.shape[0])]), tf.float32)

    return anchors


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = tf.convert_to_tensor(np.array([1, 1, base_size, base_size]) - 1.)
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = tf.cast(tf.concat([_scale_enum(ratio_anchors[i, :], scales)
                                for i in range(ratio_anchors.shape[0])], axis=0), tf.float32)
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    # ws = tf.expand_dims(ws, axis=1)
    # hs = tf.expand_dims(hs, axis=1)
    anchors = tf.stack([x_ctr - 0.5 * (ws - 1.), y_ctr - 0.5 * (hs - 1.),
              x_ctr + 0.5 * (ws - 1.), y_ctr + 0.5 * (hs - 1.)], axis=1)

    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = tf.round(tf.sqrt(size_ratios))
    hs = tf.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32),
                         anchor_ratios=(0.5, 1, 2), base_size=16):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'
    """
    anchors = generate_anchors(
        base_size=base_size, ratios=np.array(anchor_ratios),
        scales=np.array(anchor_scales))
    A = tf.shape(anchors)[0]
    shift_x = tf.range(width) * feat_stride
    shift_y = tf.range(height) * feat_stride
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.reshape(shift_x, [-1, ])
    shift_y = tf.reshape(shift_y, [-1, ])
    shifts = tf.transpose(tf.stack([shift_x, shift_y, shift_x, shift_y]))
    K = tf.shape(shifts)[0]
    # width changes faster, so here it is H, W, C
    anchors = tf.cast(tf.reshape(anchors, [1, A, 4]), tf.float32) + tf.cast(tf.transpose(tf.reshape(shifts, [1, K, 4]), [1, 0, 2]), tf.float32)
    anchors = tf.cast(tf.reshape(anchors, [K * A, 4]), tf.float32)

    return anchors


if __name__ == '__main__':
    anchors_tf = generate_anchors_pre(64, 64, 8, anchor_scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]) * 8,
                                   anchor_ratios=(0.5, 1.0, 2.0), base_size=4)
    with tf.Session() as sess:
        anchors = sess.run(anchors_tf)
        print(anchors[:9])
        print(anchors.shape)

        # x_c = (anchors[:, 2] - anchors[:, 0]) / 2
        # y_c = (anchors[:, 3] - anchors[:, 1]) / 2
        # h = anchors[:, 2] - anchors[:, 0] + 1
        # w = anchors[:, 3] - anchors[:, 1] + 1
        # theta = -90 * np.ones_like(x_c)
        # anchors = np.stack([x_c, y_c]).transpose()
        # print(anchors.shape)
