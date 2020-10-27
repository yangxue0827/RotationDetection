# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from libs.configs import cfgs
import tensorflow as tf
from libs.utils.rotate_polygon_nms import rotate_gpu_nms


def nms_rotate(decode_boxes, scores, iou_threshold, max_output_size, use_gpu=True, gpu_id=0):
    """
    :param boxes: format [x_c, y_c, w, h, theta]
    :param scores: scores of boxes
    :param threshold: iou threshold (0.7 or 0.5)
    :param max_output_size: max number of output
    :return: the remaining index of boxes
    """

    if use_gpu:
        keep = nms_rotate_gpu(boxes_list=decode_boxes,
                              scores=scores,
                              iou_threshold=iou_threshold,
                              device_id=gpu_id)

        keep = tf.cond(
            tf.greater(tf.shape(keep)[0], max_output_size),
            true_fn=lambda: tf.slice(keep, [0], [max_output_size]),
            false_fn=lambda: keep)

    else:
        keep = tf.py_func(nms_rotate_cpu,
                          inp=[decode_boxes, scores, iou_threshold, max_output_size],
                          Tout=tf.int64)
    return keep


def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):

    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue

            if np.sqrt((boxes[i, 0] - boxes[j, 0])**2 + (boxes[i, 1] - boxes[j, 1])**2) > (boxes[i, 2] + boxes[j, 2] + boxes[i, 3] + boxes[j, 3]):
                inter = 0.0
            else:

                r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
                area_r2 = boxes[j, 2] * boxes[j, 3]
                inter = 0.0

                try:
                    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                    if int_pts is not None:
                        order_pts = cv2.convexHull(int_pts, returnPoints=True)

                        int_area = cv2.contourArea(order_pts)

                        inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + cfgs.EPSILON)

                except:
                    """
                      cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                      error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                    """
                    # print(r1)
                    # print(r2)
                    inter = 0.9999

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)


def nms_rotate_gpu(boxes_list, scores, iou_threshold, device_id=0):
    x_c, y_c, w, h, theta = tf.unstack(boxes_list, axis=1)
    boxes_list = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
    det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
    keep = tf.py_func(rotate_gpu_nms,
                      inp=[det_tensor, iou_threshold, device_id],
                      Tout=tf.int64)
    keep = tf.reshape(keep, [-1])
    return keep


if __name__ == '__main__':
    boxes = np.array([[50, 50, 100, 100, 0],
                      [60, 60, 100, 100, 0],
                      [50, 50, 100, 100, -45.],
                      [200, 200, 100, 100, 0.]])

    scores = np.array([0.99, 0.88, 0.66, 0.77])

    keep = nms_rotate(tf.convert_to_tensor(boxes, dtype=tf.float32), tf.convert_to_tensor(scores, dtype=tf.float32),
                      0.7, 5)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    with tf.Session() as sess:
        print(sess.run(keep))
