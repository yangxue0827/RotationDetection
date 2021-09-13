# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

sys.path.append('../..')
# from utils.gaussian_wasserstein_distance import get_element1, get_element4
from libs.utils.coordinate_convert import *
# from libs.utils.rbbox_overlaps import rbbx_overlaps
# from libs.utils.iou_cpu import get_iou_matrix


def iou_rotate_calculate2(boxes1, boxes2):
    ious = []
    if boxes1.shape[0] != 0:
        boxes1[:, 2] += 1.0
        boxes1[:, 3] += 1.0
        boxes2[:, 2] += 1.0
        boxes2[:, 3] += 1.0

        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]

        for i in range(boxes1.shape[0]):
            temp_ious = []
            r1 = ((boxes1[i][0], boxes1[i][1]), (boxes1[i][2], boxes1[i][3]), boxes1[i][4])
            r2 = ((boxes2[i][0], boxes2[i][1]), (boxes2[i][2], boxes2[i][3]), boxes2[i][4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[i] - int_area + 1e-4)

                # if boxes1[i][2] < 0.1 or boxes1[i][3] < 0.1 or boxes2[i][2] < 0.1 or boxes2[i][3] < 0.1:
                #     inter = 0

                inter = max(0.0, min(1.0, inter))

                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
            ious.append(temp_ious)

    return np.array(ious, dtype=np.float32)


def diou_rotate_calculate(boxes1, boxes2):

    if boxes1.shape[0] != 0:
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        d = (boxes1[:, 0] - boxes2[:, 0]) ** 2 + (boxes1[:, 1] - boxes2[:, 1])

        boxes1_ = forward_convert(boxes1, with_label=False)
        boxes2_ = forward_convert(boxes2, with_label=False)

        xmin = np.minimum(np.min(boxes1_[:, 0::2]), np.min(boxes2_[:, 0::2]))
        xmax = np.maximum(np.max(boxes1_[:, 0::2]), np.max(boxes2_[:, 0::2]))
        ymin = np.minimum(np.min(boxes1_[:, 1::2]), np.min(boxes2_[:, 1::2]))
        ymax = np.maximum(np.max(boxes1_[:, 1::2]), np.max(boxes2_[:, 1::2]))

        c = (xmax - xmin) ** 2 + (ymax - ymin) ** 2
        ious = []
        for i in range(boxes1.shape[0]):
            r1 = ((boxes1[i][0], boxes1[i][1]), (boxes1[i][2], boxes1[i][3]), boxes1[i][4])
            r2 = ((boxes2[i][0], boxes2[i][1]), (boxes2[i][2], boxes2[i][3]), boxes2[i][4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                iou = int_area * 1.0 / (area1[i] + area2[i] - int_area)
            else:
                iou = 0.0

            ious.append(iou)
        ious = np.array(ious)

        dious = ious - d / c
    else:
        dious = []

    return np.reshape(np.array(dious, dtype=np.float32), [-1, 1])


def gaussian_wasserstein_distance_(boxes1, boxes2):
    boxes1 = coordinate_present_convert(boxes1, -1)
    boxes1[:, 4] += 90
    boxes1[:, 4] *= (-np.pi / 180)

    boxes2 = coordinate_present_convert(boxes2, -1)
    boxes2[:, 4] += 90
    boxes2[:, 4] *= (-np.pi / 180)

    dis = (boxes1[:, 0] - boxes2[:, 0])**2 + (boxes1[:, 1] - boxes2[:, 1])**2 + \
          ((boxes1[:, 2] / 2 * np.cos(boxes1[:, 4])**2 + boxes1[:, 3] / 2 * np.sin(boxes1[:, 4])**2) - (boxes2[:, 2] / 2 * np.cos(boxes2[:, 4])**2 + boxes2[:, 3] / 2 * np.sin(boxes2[:, 4])**2))**2 + \
          2*((boxes1[:, 2] / 2 * np.cos(boxes1[:, 4])* np.sin(boxes1[:, 4] - boxes1[:, 3] / 2 * np.cos(boxes1[:, 4]) * np.sin(boxes1[:, 4]))) - (boxes2[:, 2] / 2 * np.cos(boxes2[:, 4]) * np.sin(boxes2[:, 4] - boxes2[:, 3] / 2 * np.cos(boxes2[:, 4]) * np.sin(boxes2[:, 4]))))**2 + \
          ((boxes1[:, 2] / 2 * np.sin(boxes1[:, 4]) ** 2 + boxes1[:, 3] / 2 * np.cos(boxes1[:, 4]) ** 2) - (
          boxes2[:, 2] / 2 * np.sin(boxes2[:, 4]) ** 2 + boxes2[:, 3] / 2 * np.cos(boxes2[:, 4]) ** 2)) ** 2
    return dis


def gaussian_wasserstein_distance(boxes1, boxes2):
    from utils.gaussian_wasserstein_distance import get_element1, get_element4

    boxes1 = coordinate_present_convert(boxes1, -1)
    boxes1[:, 4] += 90
    boxes1[:, 4] *= (-np.pi / 180)
    #
    boxes2 = coordinate_present_convert(boxes2, -1)
    boxes2[:, 4] += 90
    boxes2[:, 4] *= (-np.pi / 180)

    element1 = get_element1(boxes1[:, 2], boxes1[:, 3], boxes1[:, 4], boxes2[:, 2], boxes2[:, 3], boxes2[:, 4])
    element4 = get_element4(boxes1[:, 2], boxes1[:, 3], boxes1[:, 4], boxes2[:, 2], boxes2[:, 3], boxes2[:, 4])
    dis = (boxes1[:, 0] - boxes2[:, 0])**2 + (boxes1[:, 1] - boxes2[:, 1])**2 + (element1 + element4)
    return dis


def wasserstein_diss_sigma(sigma1, sigma2):
    wasserstein_diss_item2 = tf.linalg.matmul(sigma1, sigma1) + tf.linalg.matmul(sigma2, sigma2) - 2 * tf.linalg.sqrtm(
        tf.linalg.matmul(tf.linalg.matmul(sigma1, tf.linalg.matmul(sigma2, sigma2)), sigma1))
    wasserstein_diss_item2 = tf.linalg.trace(wasserstein_diss_item2)
    return wasserstein_diss_item2


def gwd(boxes1, boxes2):
    x1, y1, w1, h1, theta1 = tf.unstack(boxes1, axis=1)
    x2, y2, w2, h2, theta2 = tf.unstack(boxes2, axis=1)
    x1 = tf.reshape(x1, [-1, 1])
    y1 = tf.reshape(y1, [-1, 1])
    h1 = tf.reshape(h1, [-1, 1])
    w1 = tf.reshape(w1, [-1, 1])
    theta1 = tf.reshape(theta1, [-1, 1])
    x2 = tf.reshape(x2, [-1, 1])
    y2 = tf.reshape(y2, [-1, 1])
    h2 = tf.reshape(h2, [-1, 1])
    w2 = tf.reshape(w2, [-1, 1])
    theta2 = tf.reshape(theta2, [-1, 1])
    theta1 *= (np.pi / 180)
    theta2 *= (np.pi / 180)

    sigma1_1 = w1 / 2 * tf.cos(theta1) ** 2 + h1 / 2 * tf.sin(theta1) ** 2
    sigma1_2 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
    sigma1_3 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
    sigma1_4 = w1 / 2 * tf.sin(theta1) ** 2 + h1 / 2 * tf.cos(theta1) ** 2
    sigma1 = tf.reshape(tf.concat([sigma1_1, sigma1_2, sigma1_3, sigma1_4], axis=-1), [-1, 2, 2])

    sigma2_1 = w2 / 2 * tf.cos(theta2) ** 2 + h2 / 2 * tf.sin(theta2) ** 2
    sigma2_2 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
    sigma2_3 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
    sigma2_4 = w2 / 2 * tf.sin(theta2) ** 2 + h2 / 2 * tf.cos(theta2) ** 2
    sigma2 = tf.reshape(tf.concat([sigma2_1, sigma2_2, sigma2_3, sigma2_4], axis=-1), [-1, 2, 2])

    wasserstein_diss_item1 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    wasserstein_diss_item2 = tf.reshape(wasserstein_diss_sigma(sigma1, sigma2), [-1, 1])
    wasserstein_diss = wasserstein_diss_item1 + wasserstein_diss_item2
    return sigma1, sigma2, wasserstein_diss


def sigma(a, w, h):
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    sig = np.array([[w/2, 0], [0, h/2]])
    res = np.dot(R, sig)
    res = np.dot(res, R.T)
    return res


if __name__ == '__main__':
    from utils.gaussian_wasserstein_distance import get_element1, get_element4

    boxes1 = np.array([[50, 50, 10, 70, -30],
                       [50, 50, 100, 700, -30]], np.float32)

    boxes2 = np.array([[10, 40, 10, 70, -30],
                       [10, 40, 100, 700, -40]], np.float32)

    # boxes1 = np.array([    # prediction box
    #     [50, 50, 10, 70, -35],     # 90 <--> 180
    #     [50, 50, 70, 10, -90.5],   # 90   PoA + EoE
    #     [50, 50, 70, 10, -90.5],   # 180  PoA
    #     [50, 50, 40, 40, -35],     # 180  w=h
    # ], np.float32)
    #
    # boxes2 = np.array([    # ground truth
    #     [50, 50, 70, 10, 55],
    #     [50, 50, 10, 70, -0.5],
    #     [50, 50, 70, 10, 89.5],
    #     [50, 50, 40, 40, 55],
    # ], np.float32)

    print('iou', iou_rotate_calculate2(boxes1, boxes2).reshape(-1,))  # [0.9999996 0.9999998 0.9999998 1.       ]
    # print(diou_rotate_calculate(boxes1, boxes2).reshape(-1,))  # [0.9999997  0.99999994 0.99999994 1.        ]
    # print(gaussian_wasserstein_distance(boxes1, boxes2))     # [6.1035156e-05 3.1062821e-04 3.1062821e-04 0.0000000e+00]


    # tmp = np.maximum(np.log(gaussian_wasserstein_distance(boxes1, boxes2)+1e-3), 0)
    # print(np.log(gaussian_wasserstein_distance(boxes1, boxes2)))
    # print(tmp/(1+tmp))

    # print(np.argsort(iou_rotate_calculate2(boxes1, boxes2).reshape(-1, )*-1))
    # print(np.argsort(diou_rotate_calculate(boxes1, boxes2).reshape(-1, )*-1))
    # print(np.argsort(np.array(gaussian_wasserstein_distance(boxes1, boxes2))))
    #
    # # print(sigma(-np.pi*35/180, 10, 70))
    # # print(sigma(np.pi*(90-35)/180, 70, 10))
    #
    sigma1_tf1, sigma2_tf1, gwd_tf1 = gwd(coordinate_present_convert(boxes1, -1, False), coordinate_present_convert(boxes2, -1, False))
    sigma1_tf2, sigma2_tf2, gwd_tf2 = gwd(boxes1, boxes2)
    # sigma1_tf2 = tf.linalg.eigvalsh(sigma1_tf2)

    with tf.Session() as sess:
        sigma1_tf_1, sigma2_tf_1, gwd_tf_1 = sess.run([sigma1_tf1, sigma2_tf1, gwd_tf1])
        sigma1_tf_2, sigma2_tf_2, gwd_tf_2 = sess.run([sigma1_tf2, sigma2_tf2, gwd_tf2])
        # print(sigma1_tf_1)
        # print(sigma2_tf_1)
        # print('**'*10)
        print(sigma1_tf_2)
        # print(sigma2_tf_2)

        print('**' * 10)
        # print(np.reshape(gwd_tf_1, [-1, ]))
        # print(np.argsort(np.reshape(gwd_tf_1, [-1, ])))

        print('gwd', np.reshape(gwd_tf_2, [-1, ]))
        print(1-1/(np.sqrt(gwd_tf_2)+1))
        # print(np.argsort(np.reshape(gwd_tf_2, [-1, ])))

        # gwd_tf_2 = np.maximum(np.log(gwd_tf_2 + 1e-3), 0.0)
        # gwd_tf_2_ = np.maximum(np.log(gwd_tf_2 + 1e-3), 0.0)
        # print(gwd_tf_2_/(5+gwd_tf_2_))
        #
        # gwd_tf_2 = np.maximum(np.log(gwd_tf_2 + 1e-3), 0.0)
        # print(gwd_tf_2)
        # print(1-1 / (5 + gwd_tf_2))
        # print(gwd_tf_2_ / (2 + gwd_tf_2_))
        # print(gwd_tf_2_ / (3 + gwd_tf_2_))
        # print(gwd_tf_2_ / (5 + gwd_tf_2_))







