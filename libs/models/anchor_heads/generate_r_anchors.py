# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import tensorflow as tf
import cv2


def make_anchors(base_anchor_size, anchor_scales, anchor_ratios, anchor_angles,
                 featuremap_height, featuremap_width, stride, name='make_ratate_anchors'):


    '''
    :param base_anchor_size:
    :param anchor_scales:
    :param anchor_ratios:
    :param anchor_thetas:
    :param featuremap_height:
    :param featuremap_width:
    :param stride:
    :return:
    '''
    with tf.variable_scope(name):
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)  # [y_center, x_center, h, w]
        ws, hs, angles = enum_ratios_and_thetas(enum_scales(base_anchor, anchor_scales),
                                                anchor_ratios, anchor_angles)  # per locations ws and hs and thetas

        x_centers = tf.range(featuremap_width, dtype=tf.float32) * stride + stride // 2
        y_centers = tf.range(featuremap_height, dtype=tf.float32) * stride + stride // 2

        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        angles, _ = tf.meshgrid(angles, x_centers)
        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)

        anchor_centers = tf.stack([x_centers, y_centers], 2)
        anchor_centers = tf.reshape(anchor_centers, [-1, 2])

        box_parameters = tf.stack([ws, hs, angles], axis=2)
        box_parameters = tf.reshape(box_parameters, [-1, 3])
        anchors = tf.concat([anchor_centers, box_parameters], axis=1)

        return anchors


def enum_scales(base_anchor, anchor_scales):
    anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))

    return anchor_scales


def enum_ratios_and_thetas(anchors, anchor_ratios, anchor_angles):
    '''
    ratio = h /w
    :param anchors:
    :param anchor_ratios:
    :return:
    '''
    ws = anchors[:, 2]  # for base anchor: w == h
    hs = anchors[:, 3]
    anchor_angles = tf.constant(anchor_angles, tf.float32)
    sqrt_ratios = tf.sqrt(tf.constant(anchor_ratios))

    ws = tf.reshape(ws / sqrt_ratios[:, tf.newaxis], [-1])
    hs = tf.reshape(hs * sqrt_ratios[:, tf.newaxis], [-1])

    ws, _ = tf.meshgrid(ws, anchor_angles)
    hs, anchor_angles = tf.meshgrid(hs, anchor_angles)

    anchor_angles = tf.reshape(anchor_angles, [-1, 1])
    ws = tf.reshape(ws, [-1, 1])
    hs = tf.reshape(hs, [-1, 1])

    return ws, hs, anchor_angles


if __name__ == '__main__':
    import os
    from libs.configs import cfgs
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    from libs.utils.show_box_in_tensor import DrawBoxTensor
    drawer = DrawBoxTensor(cfgs)

    base_anchor_size = 256
    anchor_scales = [1.]
    anchor_ratios = [0.5, 2.0, 1/3, 3, 1/5, 5, 1/8, 8]
    anchor_angles = [-90, -75, -60, -45, -30, -15]
    base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)
    tmp1 = enum_ratios_and_thetas(enum_scales(base_anchor, anchor_scales), anchor_ratios, anchor_angles)
    anchors = make_anchors(32,
                           [2.], [2.0, 1/2], anchor_angles,
                           featuremap_height=800 // 8,
                           featuremap_width=800 // 8,
                           stride=8)

    # anchors = make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
    #                        anchor_scales=cfgs.ANCHOR_SCALES,
    #                        anchor_ratios=cfgs.ANCHOR_RATIOS,
    #                        anchor_angles=cfgs.ANCHOR_ANGLES,
    #                        featuremap_height=800 // 16,
    #                        featuremap_width=800 // 16,
    #                        stride=cfgs.ANCHOR_STRIDE[0],
    #                        name="make_anchors_forRPN")

    img = tf.zeros([800, 800, 3])
    img = tf.expand_dims(img, axis=0)

    img1 = drawer.only_draw_boxes(img, anchors[9100:9110], 'r')

    with tf.Session() as sess:
        temp1, _img1 = sess.run([anchors, img1])

        _img1 = _img1[0]

        cv2.imwrite('rotate_anchors.jpg', _img1)
        cv2.waitKey(0)

        print(temp1)
        print('debug')
