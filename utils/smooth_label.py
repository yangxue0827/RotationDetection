# -*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function
import numpy as np
import math


def gaussian_label(label, num_class, u=0, sig=4.0):
    x = np.array(range(math.floor(-num_class/2), math.ceil(num_class/2), 1))
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    return np.concatenate([y_sig[math.ceil(num_class/2)-label:],
                           y_sig[:math.ceil(num_class/2)-label]], axis=0)


def rectangular_label(label, num_class, raduius=4):
    x = np.zeros([num_class])
    x[:raduius+1] = 1
    x[-raduius:] = 1
    y_sig = np.concatenate([x[-label:], x[:-label]], axis=0)
    return y_sig


def pulse_label(label, num_class):
    x = np.zeros([num_class])
    x[label] = 1
    return x


def triangle_label(label, num_class, raduius=4):
    y_sig = np.zeros([num_class])
    x = np.array(range(raduius+1))
    y = -1/(raduius+1) * x + 1
    y_sig[:raduius+1] = y
    y_sig[-raduius:] = y[-1:0:-1]

    return np.concatenate([y_sig[-label:], y_sig[:-label]], axis=0)


def get_all_smooth_label(num_label, label_type=0, raduius=4):
    all_smooth_label = []

    if label_type == 0:
        for i in range(num_label):
            all_smooth_label.append(gaussian_label(i, num_label, sig=raduius))
    elif label_type == 1:
        for i in range(num_label):
            all_smooth_label.append(rectangular_label(i, num_label, raduius=raduius))
    elif label_type == 2:
        for i in range(num_label):
            all_smooth_label.append(pulse_label(i, num_label))
    elif label_type == 3:
        for i in range(num_label):
            all_smooth_label.append(triangle_label(i, num_label, raduius=raduius))
    else:
        raise Exception('Only support gaussian, rectangular, triangle and pulse label')
    return np.array(all_smooth_label)


def angle_smooth_label(angle_label, angle_range=90, label_type=0, raduius=4, omega=1):
    """
    :param angle_label: [-90,0) or [-90, 0)
    :param angle_range: 90 or 180
    :return:
    """

    assert angle_range % omega == 0, 'wrong omega'

    angle_range /= omega
    angle_label /= omega

    angle_label = np.array(-np.round(angle_label), np.int32)
    all_smooth_label = get_all_smooth_label(int(angle_range), label_type, raduius)
    inx = angle_label == angle_range
    angle_label[inx] = angle_range - 1
    smooth_label = all_smooth_label[angle_label]
    return np.array(smooth_label, np.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # angle_label = np.array([-89.9, -45.2, -0.3, -1.9])
    # smooth_label = angle_smooth_label(angle_label)

    # y_sig = triangle_label(30, 180, raduius=8)
    # y_sig = gaussian_label(30, 180, sig=0.1)
    # y_sig = pulse_label(30, 180)
    y_sig = triangle_label(0, 90)
    x = np.array(range(0, 90, 1))
    plt.plot(x, y_sig, "r-", linewidth=2)
    plt.grid(True)
    plt.show()
    print(y_sig)
