# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np


def iou_calculate(boxes_1, boxes_2):

    with tf.name_scope('iou_caculate'):

        xmin_1, ymin_1, xmax_1, ymax_1 = tf.unstack(boxes_1, axis=1)  # ymin_1 shape is [N, 1]..

        xmin_2, ymin_2, xmax_2, ymax_2 = tf.unstack(boxes_2, axis=1)  # ymin_2 shape is [M, ]..

        max_xmin = tf.maximum(xmin_1, xmin_2)
        min_xmax = tf.minimum(xmax_1, xmax_2)

        max_ymin = tf.maximum(ymin_1, ymin_2)
        min_ymax = tf.minimum(ymax_1, ymax_2)

        overlap_h = tf.maximum(0., min_ymax - max_ymin)  # avoid h < 0
        overlap_w = tf.maximum(0., min_xmax - max_xmin)

        overlaps = overlap_h * overlap_w

        area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
        area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

        iou = overlaps / (area_1 + area_2 - overlaps)

        return iou


def iou_calculate_np(boxes_1, boxes_2):
    xmin_1, ymin_1, xmax_1, ymax_1 = np.split(boxes_1, 4, axis=1)
    # xmin_1, ymin_1, xmax_1, ymax_1 = boxes_1[:, 0], boxes_1[:, 1], boxes_1[:, 2], boxes_1[:, 3]

    xmin_2, ymin_2, xmax_2, ymax_2 = boxes_2[:, 0], boxes_2[:, 1], boxes_2[:, 2], boxes_2[:, 3]

    max_xmin = np.maximum(xmin_1, xmin_2)
    min_xmax = np.minimum(xmax_1, xmax_2)

    max_ymin = np.maximum(ymin_1, ymin_2)
    min_ymax = np.minimum(ymax_1, ymax_2)

    overlap_h = np.maximum(0., min_ymax - max_ymin)  # avoid h < 0
    overlap_w = np.maximum(0., min_xmax - max_xmin)

    overlaps = overlap_h * overlap_w

    area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
    area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

    iou = overlaps / (area_1 + area_2 - overlaps)

    return iou


def iou_calculate1(boxes_1, boxes_2):

    xmin_1, ymin_1, xmax_1, ymax_1 = boxes_1[:, 0], boxes_1[:, 1], boxes_1[:, 2], boxes_1[:, 3]

    xmin_2, ymin_2, xmax_2, ymax_2 = boxes_2[:, 0], boxes_2[:, 1], boxes_2[:, 2], boxes_2[:, 3]

    max_xmin = np.maximum(xmin_1, xmin_2)
    min_xmax = np.minimum(xmax_1, xmax_2)

    max_ymin = np.maximum(ymin_1, ymin_2)
    min_ymax = np.minimum(ymax_1, ymax_2)

    overlap_h = np.maximum(0., min_ymax - max_ymin)  # avoid h < 0
    overlap_w = np.maximum(0., min_xmax - max_xmin)

    overlaps = overlap_h * overlap_w

    area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
    area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

    iou = overlaps / (area_1 + area_2 - overlaps)

    return iou


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '13'
    boxes1 = np.array([[50, 50, 100, 300],
                       [60, 60, 100, 200]], np.float32)

    boxes2 = np.array([[50, 50, 100, 300],
                       [200, 200, 100, 200]], np.float32)

    print(iou_calculate_np(boxes1, boxes2))



