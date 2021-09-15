# -*- coding:utf-8 -*-
# Author: Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>

# License: Apache-2.0 license

from __future__ import absolute_import, division, print_function
import numpy as np
import math


def gaussian_label(label, num_class, u=0, sig=4.0):
    """
    Get gaussian label

    :param label:  angle_label/omega
    :param num_class: angle_range/omega
    :param u: mean
    :param sig: window radius
    :return: gaussian label
    """
    x = np.array(range(math.floor(-num_class/2), math.ceil(num_class/2), 1))
    if num_class % 2 != 0:
        x = x[:-1]
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    return np.concatenate([y_sig[math.ceil(num_class/2)-label:],
                           y_sig[:math.ceil(num_class/2)-label]], axis=0)


def rectangular_label(label, num_class, radius=4):
    """
    Get rectangular label

    :param label: angle_label/omega
    :param num_class: angle_range/omega
    :param radius: window radius
    :return: rectangular label
    """
    x = np.zeros([num_class])
    x[:radius+1] = 1
    x[-radius:] = 1
    y_sig = np.concatenate([x[-label:], x[:-label]], axis=0)
    return y_sig


def pulse_label(label, num_class):
    """
    Get pulse label

    :param label: angle_label/omega
    :param num_class: angle_range/omega
    :return: pulse label
    """
    x = np.zeros([num_class])
    x[label] = 1
    return x


def triangle_label(label, num_class, radius=4):
    """
    Get triangle label

    :param label: angle_label/omega
    :param num_class: angle_range/omega
    :param radius: window radius
    :return: triangle label
    """
    y_sig = np.zeros([num_class])
    x = np.array(range(radius+1))
    y = -1/(radius+1) * x + 1
    y_sig[:radius+1] = y
    y_sig[-radius:] = y[-1:0:-1]

    return np.concatenate([y_sig[-label:], y_sig[:-label]], axis=0)


def get_all_smooth_label(num_label, label_type=0, radius=4):
    all_smooth_label = []

    if label_type == 0:
        for i in range(num_label):
            all_smooth_label.append(gaussian_label(i, num_label, sig=radius))
    elif label_type == 1:
        for i in range(num_label):
            all_smooth_label.append(rectangular_label(i, num_label, radius=radius))
    elif label_type == 2:
        for i in range(num_label):
            all_smooth_label.append(pulse_label(i, num_label))
    elif label_type == 3:
        for i in range(num_label):
            all_smooth_label.append(triangle_label(i, num_label, radius=radius))
    else:
        raise Exception('Only support gaussian, rectangular, triangle and pulse label')
    return np.array(all_smooth_label)


def angle_smooth_label(angle_label, angle_range=90, label_type=0, radius=4, omega=1):
    """

    :param angle_label: angle label, range in [-90,0) or [-180, 0)
    :param angle_range: 90 or 180
    :param label_type: 0: gaussian label, 1: rectangular label, 2: pulse label, 3: triangle label
    :param radius: window radius
    :param omega: angle discretization granularity
    :return:

    **Circular Smooth Label:**
    Proposed by `"Xue Yang et al. Arbitrary-Oriented Object Detection with Circular Smooth Label. ECCV 2020."
    <https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40>`_

    .. image:: ../../images/csl.jpg
    """

    assert angle_range % omega == 0, 'wrong omega'

    angle_range /= omega
    angle_label /= omega

    angle_label = np.array(-np.round(angle_label), np.int32)
    all_smooth_label = get_all_smooth_label(int(angle_range), label_type, radius)
    inx = angle_label == angle_range
    angle_label[inx] = angle_range - 1
    smooth_label = all_smooth_label[angle_label]
    return np.array(smooth_label, np.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # angle_label = np.array([-89.9, -45.2, -0.3, -1.9])
    # smooth_label = angle_smooth_label(angle_label)
    # y_sig = triangle_label(30, 180, radius=8)
    y_sig = gaussian_label(3, 180, sig=6)
    # y_sig = pulse_label(40, 180)
    # y_sig = triangle_label(3, 180, radius=1)
    # x = np.array(range(0, 180, 1))
    # plt.plot(x, y_sig, "r-", linewidth=2)
    # plt.grid(True)
    # plt.show()
    # print(y_sig)
    # print(y_sig.shape)

    import tensorflow as tf

    y_sig = tf.convert_to_tensor(y_sig, tf.float32)
    y_sig_dct = tf.signal.dct(tf.reshape(y_sig, [-1, 180]),
                              type=3, n=8, axis=-1)
    y_sig_idct = tf.signal.idct(tf.reshape(y_sig_dct, [-1, 8]),
                                type=3, n=180, axis=-1)
    with tf.Session() as sess:
        y_sig_, y_sig_dct_, y_sig_idct_ = sess.run([y_sig, y_sig_dct, y_sig_idct])
        print(y_sig_)
        print(np.argmax(y_sig_, axis=-1))
        print(y_sig_dct_)
        print(y_sig_idct_)
        print(np.argmax(y_sig_idct_, axis=-1))


