# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def wasserstein_distance_sigma(sigma1, sigma2):

    """
    :math: `\mathbf{Tr}\left(\mathbf{\Sigma}_{1}+\mathbf{\Sigma}_{2}-2(\mathbf{\Sigma}_{1}^{1/2}\mathbf{\Sigma}_{2}\mathbf{\Sigma}_{1}^{1/2})^{1/2}\right)`

    :param sigma1:
    :param sigma2:
    :return:
    """

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
    wasserstein_diss_item2 = tf.reshape(wasserstein_distance_sigma(sigma1, sigma2), [-1, 1])
    wasserstein_diss = wasserstein_diss_item1 + wasserstein_diss_item2
    return sigma1, sigma2, wasserstein_diss
