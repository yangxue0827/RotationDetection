# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np


def bbox_transform_inv(boxes, deltas, scale_factors=None):
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    if scale_factors:
        dx /= scale_factors[0]
        dy /= scale_factors[1]
        dw /= scale_factors[2]
        dh /= scale_factors[3]

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = tf.exp(dw) * widths
    pred_h = tf.exp(dh) * heights

    predict_xmin = pred_ctr_x - 0.5 * pred_w
    predict_xmax = pred_ctr_x + 0.5 * pred_w
    predict_ymin = pred_ctr_y - 0.5 * pred_h
    predict_ymax = pred_ctr_y + 0.5 * pred_h

    return tf.transpose(tf.stack([predict_xmin, predict_ymin,
                                  predict_xmax, predict_ymax]))


def bbox_transform(ex_rois, gt_rois, scale_factors=None):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths + 1e-5)
    targets_dh = np.log(gt_heights / ex_heights + 1e-5)

    if scale_factors:
        targets_dx *= scale_factors[0]
        targets_dy *= scale_factors[1]
        targets_dw *= scale_factors[2]
        targets_dh *= scale_factors[3]

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets


def rbbox_transform_inv(boxes, deltas, scale_factors=None):
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    dtheta = deltas[:, 4]

    if scale_factors:
        dx /= scale_factors[0]
        dy /= scale_factors[1]
        dw /= scale_factors[2]
        dh /= scale_factors[3]
        dtheta /= scale_factors[4]

    # BBOX_XFORM_CLIP = tf.log(cfgs.IMG_SHORT_SIDE_LEN / 16.)
    # dw = tf.minimum(dw, BBOX_XFORM_CLIP)
    # dh = tf.minimum(dh, BBOX_XFORM_CLIP)

    pred_ctr_x = dx * boxes[:, 2] + boxes[:, 0]
    pred_ctr_y = dy * boxes[:, 3] + boxes[:, 1]
    pred_w = tf.exp(dw) * boxes[:, 2]
    pred_h = tf.exp(dh) * boxes[:, 3]

    pred_theta = dtheta * 180 / np.pi + boxes[:, 4]

    return tf.transpose(tf.stack([pred_ctr_x, pred_ctr_y,
                                  pred_w, pred_h, pred_theta]))


def rbbox_transform_inv_dcl(boxes, deltas, scale_factors=None):
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    if scale_factors:
        dx /= scale_factors[0]
        dy /= scale_factors[1]
        dw /= scale_factors[2]
        dh /= scale_factors[3]

    # BBOX_XFORM_CLIP = tf.log(cfgs.IMG_SHORT_SIDE_LEN / 16.)
    # dw = tf.minimum(dw, BBOX_XFORM_CLIP)
    # dh = tf.minimum(dh, BBOX_XFORM_CLIP)

    pred_ctr_x = dx * boxes[:, 2] + boxes[:, 0]
    pred_ctr_y = dy * boxes[:, 3] + boxes[:, 1]
    pred_w = tf.exp(dw) * boxes[:, 2]
    pred_h = tf.exp(dh) * boxes[:, 3]

    return tf.transpose(tf.stack([pred_ctr_x, pred_ctr_y, pred_w, pred_h]))


def rbbox_transform(ex_rois, gt_rois, scale_factors=None):

    targets_dx = (gt_rois[:, 0] - ex_rois[:, 0]) / (ex_rois[:, 2] + 1)
    targets_dy = (gt_rois[:, 1] - ex_rois[:, 1]) / (ex_rois[:, 3] + 1)

    targets_dw = np.log(gt_rois[:, 2] / (ex_rois[:, 2] + 1) + 1e-5)
    targets_dh = np.log(gt_rois[:, 3] / (ex_rois[:, 3] + 1) + 1e-5)

    targets_dtheta = (gt_rois[:, 4] - ex_rois[:, 4]) * np.pi / 180

    if scale_factors:
        targets_dx *= scale_factors[0]
        targets_dy *= scale_factors[1]
        targets_dw *= scale_factors[2]
        targets_dh *= scale_factors[3]
        targets_dtheta *= scale_factors[4]

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh, targets_dtheta)).transpose()

    return targets


def qbbox_transform(ex_rois, gt_rois, scale_factors=None):

    w = ex_rois[:, 8]
    h = ex_rois[:, 9]

    targets_dx1 = (gt_rois[:, 0] - ex_rois[:, 0]) / w
    targets_dy1 = (gt_rois[:, 1] - ex_rois[:, 1]) / h
    targets_dx2 = (gt_rois[:, 2] - ex_rois[:, 2]) / w
    targets_dy2 = (gt_rois[:, 3] - ex_rois[:, 3]) / h
    targets_dx3 = (gt_rois[:, 4] - ex_rois[:, 4]) / w
    targets_dy3 = (gt_rois[:, 5] - ex_rois[:, 5]) / h
    targets_dx4 = (gt_rois[:, 6] - ex_rois[:, 6]) / w
    targets_dy4 = (gt_rois[:, 7] - ex_rois[:, 7]) / h

    # if scale_factors:
    #     targets_dx *= scale_factors[0]
    #     targets_dy *= scale_factors[1]
    #     targets_dw *= scale_factors[2]
    #     targets_dh *= scale_factors[3]
    #     targets_dtheta *= scale_factors[4]

    targets = np.vstack((targets_dx1, targets_dy1, targets_dx2, targets_dy2,
                         targets_dx3, targets_dy3, targets_dx4, targets_dy4)).transpose()

    return targets


def qbbox_transform_inv(boxes, deltas, scale_factors=None):
    dx1 = deltas[:, 0]
    dy1 = deltas[:, 1]
    dx2 = deltas[:, 2]
    dy2 = deltas[:, 3]
    dx3 = deltas[:, 4]
    dy3 = deltas[:, 5]
    dx4 = deltas[:, 6]
    dy4 = deltas[:, 7]

    # if scale_factors:
    #     dx /= scale_factors[0]
    #     dy /= scale_factors[1]
    #     dw /= scale_factors[2]
    #     dh /= scale_factors[3]
    #     dtheta /= scale_factors[4]

    pred_x_1 = dx1 * boxes[:, 2] + boxes[:, 0]
    pred_y_1 = dy1 * boxes[:, 3] + boxes[:, 1]
    pred_x_2 = dx2 * boxes[:, 2] + boxes[:, 0]
    pred_y_2 = dy2 * boxes[:, 3] + boxes[:, 1]
    pred_x_3 = dx3 * boxes[:, 2] + boxes[:, 0]
    pred_y_3 = dy3 * boxes[:, 3] + boxes[:, 1]
    pred_x_4 = dx4 * boxes[:, 2] + boxes[:, 0]
    pred_y_4 = dy4 * boxes[:, 3] + boxes[:, 1]

    # pred_theta = dtheta * 180 / np.pi + boxes[:, 4]

    return tf.transpose(tf.stack([pred_x_1, pred_y_1, pred_x_2, pred_y_2, pred_x_3, pred_y_3, pred_x_4, pred_y_4]))


def qbbox_transform_inv_(boxes, deltas, scale_factors=None):
    dx1 = deltas[:, 0]
    dy1 = deltas[:, 1]
    dx2 = deltas[:, 2]
    dy2 = deltas[:, 3]
    dx3 = deltas[:, 4]
    dy3 = deltas[:, 5]
    dx4 = deltas[:, 6]
    dy4 = deltas[:, 7]

    w= boxes[:, 2] - boxes[:, 0] + 1
    h= boxes[:, 3] - boxes[:, 1] + 1

    x1=boxes[:, 0]
    y1=boxes[:, 1]
    x2=boxes[:, 0]
    y2=boxes[:, 1]+h
    x3=boxes[:, 0]+w
    y3=boxes[:, 1]+h
    x4=boxes[:, 0]+w
    y4=boxes[:, 1]

    # if scale_factors:
    #     dx /= scale_factors[0]
    #     dy /= scale_factors[1]
    #     dw /= scale_factors[2]
    #     dh /= scale_factors[3]
    #     dtheta /= scale_factors[4]

    pred_x_1 = dx1 * w + x1
    pred_y_1 = dy1 * h + y1
    pred_x_2 = dx2 * w + x2
    pred_y_2 = dy2 * h + y2
    pred_x_3 = dx3 * w + x3
    pred_y_3 = dy3 * h + y3
    pred_x_4 = dx4 * w + x4
    pred_y_4 = dy4 * h + y4

    # pred_theta = dtheta * 180 / np.pi + boxes[:, 4]

    return tf.transpose(tf.stack([pred_x_1, pred_y_1,pred_x_2, pred_y_2,pred_x_3, pred_y_3,pred_x_4, pred_y_4]))


def poly_transform_inv(boxes, deltas, point_num):
    pred_xy = []
    for i in range(point_num):
        dx, dy = deltas[:, 2 * i], deltas[:, 2 * i + 1]
        pred_x = dx * boxes[:, 2] + boxes[:, 0]
        pred_y = dy * boxes[:, 3] + boxes[:, 1]
        pred_xy.append(pred_x)
        pred_xy.append(pred_y)

    return tf.transpose(tf.stack(pred_xy))
