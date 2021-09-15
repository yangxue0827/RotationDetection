# -*- coding:utf-8 -*-

# Author: Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>
#         Liping Hou <houliping17@mails.ucas.ac.cn>
#         yue Zhou <sjtu_zy@sjtu.edu.cn>
# License: Apache-2.0 license

from __future__ import absolute_import, division, print_function
import numpy as np
import math
import sys

sys.path.append('../')


def get_code_len(class_range, mode=0):
    """
    Get encode length

    :param class_range: angle_range/omega
    :param mode: 0: binary label, 1: gray label
    :return: encode length
    """
    if mode in [0, 1]:
        return math.ceil(math.log(class_range, 2))
    else:
        raise Exception('Only support binary, gray coded label')


def get_all_binary_label(num_label, class_range):
    """
    Get all binary label according to num_label

    :param num_label: angle_range/omega, 90/omega or 180/omega
    :param class_range: angle_range/omega, 90/omega or 180/omega
    :return: all binary label
    """
    all_binary_label = []
    coding_len = get_code_len(class_range)
    tmp = 10 ** coding_len
    for i in range(num_label):
        binay = bin(i)
        binay = int(binay.split('0b')[-1]) + tmp
        binay = np.array(list(str(binay)[1:]), np.int32)
        all_binary_label.append(binay)
    return np.array(all_binary_label)


def binary_label_encode(angle_label, angle_range, omega=1.):
    """
    Encode angle label as binary label

    :param angle_label: angle label, range in [-90,0) or [-180, 0)
    :param angle_range: 90 or 180
    :param omega: angle discretization granularity
    :return: binary label
    """

    assert (angle_range / omega) % 1 == 0, 'wrong omega'

    angle_label = np.array(angle_label, np.int32)
    angle_label = np.divide(angle_label, omega)
    angle_range /= omega
    angle_range = int(angle_range)

    angle_label = np.array(-np.round(angle_label), np.int32)
    inx = angle_label == angle_range
    angle_label[inx] = 0
    all_binary_label = get_all_binary_label(angle_range, angle_range)
    binary_label = all_binary_label[angle_label]
    return np.array(binary_label, np.float32)


def binary_label_decode(binary_label, angle_range, omega=1.):
    """
    Decode binary label back to angle label

    :param binary_label: binary label
    :param angle_range: 90 or 180
    :param omega: angle discretization granularity
    :return: angle label
    """
    angle_range /= omega
    angle_range = int(angle_range)
    angle_label = np.array(np.round(binary_label), np.int32)
    angle_label = angle_label.tolist()
    all_angle_label = []
    str_angle = ''
    for i in angle_label:
        decode_angle_label = int(str_angle.join(map(str, i)), 2)
        decode_angle_label = angle_range if decode_angle_label == 0 else decode_angle_label
        decode_angle_label = decode_angle_label \
            if 0 < decode_angle_label <= int(angle_range) \
            else decode_angle_label - int(angle_range / 2)
        all_angle_label.append(decode_angle_label * omega)
    return np.array(all_angle_label, np.float32)


def get_all_gray_label(angle_range):
    """
    Get all gray label

    :param angle_range: 90/omega or 180/omega
    :return: all gray label
    """
    coding_len = get_code_len(angle_range)
    return np.array(get_grace(['0', '1'], 1, coding_len))


def get_grace(list_grace, n, maxn):

    if n >= maxn:
        return list_grace
    list_befor, list_after = [], []
    for i in range(len(list_grace)):
        list_befor.append('0' + list_grace[i])
        list_after.append('1' + list_grace[-(i + 1)])
    return get_grace(list_befor + list_after, n + 1, maxn)


def gray_label_encode(angle_label, angle_range, omega=1.):
    """
    Encode angle label as gray label

    :param angle_label: angle label, range in [-90,0) or [-180, 0)
    :param angle_range: 90 or 180
    :param omega: angle discretization granularity
    :return: gray label
    """

    assert (angle_range / omega) % 1 == 0, 'wrong omega'

    angle_label = np.array(angle_label, np.int32)
    angle_label = np.divide(angle_label, omega)
    angle_range /= omega
    angle_range = int(angle_range)

    angle_label = np.array(-np.round(angle_label), np.int32)
    inx = angle_label == angle_range
    angle_label[inx] = 0
    all_gray_label = get_all_gray_label(angle_range)
    gray_label = all_gray_label[angle_label]
    return np.array([list(map(int, ''.join(a))) for a in gray_label], np.float32)


def gray_label_decode(gray_label, angle_range, omega=1.):
    """
    Decode gray label back to angle label

    :param gray_label: gray label
    :param angle_range: 90 or 180
    :param omega: angle discretization granularity
    :return: angle label
    """
    angle_range /= omega
    angle_range = int(angle_range)
    angle_label = np.array(np.round(gray_label), np.int32)
    angle_label = angle_label.tolist()
    all_angle_label = []
    all_gray_label = list(get_all_gray_label(angle_range))
    str_angle = ''
    for i in angle_label:
        decode_angle_label = all_gray_label.index(str_angle.join(map(str, i)))
        decode_angle_label = angle_range if decode_angle_label == 0 else decode_angle_label
        decode_angle_label = decode_angle_label \
            if 0 < decode_angle_label <= int(angle_range) \
            else decode_angle_label - int(angle_range / 2)
        all_angle_label.append(decode_angle_label * omega)
    return np.array(all_angle_label, np.float32)


def angle_label_encode(angle_label, angle_range, omega=1., mode=0):

    """
    Encode angle label as binary/gray label

    :param angle_label: angle label, range in [-90,0) or [-180, 0)
    :param angle_range: 90 or 180
    :param omega: angle discretization granularity
    :param mode: 0: binary label, 1: gray label
    :return: binary/gray label

    **Dense Coded Label:**
    Proposed by `"Xue Yang et al. Dense Label Encoding for Boundary Discontinuity Free Rotation Detection. CVPR 2021."
    <https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Dense_Label_Encoding_for_Boundary_Discontinuity_Free_Rotation_Detection_CVPR_2021_paper.pdf>`_

    .. image:: ../../images/dcl_1.png
    .. image:: ../../images/dcl_2.png
    """

    if mode == 0:
        angle_binary_label = binary_label_encode(angle_label, angle_range, omega=omega)
        return angle_binary_label
    elif mode == 1:
        angle_gray_label = gray_label_encode(angle_label, angle_range, omega=omega)
        return angle_gray_label
    else:
        raise Exception('Only support binary, gray and dichotomy coded label')


def angle_label_decode(angle_encode_label, angle_range, omega=1., mode=0):

    """
    Decode binary/gray label back to angle label

    :param angle_encode_label: binary/gray label
    :param angle_label: angle label, range in [-90,0) or [-180, 0)
    :param angle_range: 90 or 180
    :param mode: 0: binary label, 1: gray label
    :return: angle label

    """
    if mode == 0:
        angle_label = binary_label_decode(angle_encode_label, angle_range, omega=omega)
    elif mode == 1:
        angle_label = gray_label_decode(angle_encode_label, angle_range, omega=omega)
    else:
        raise Exception('Only support binary, gray and dichotomy coded label')
    return angle_label


if __name__ == '__main__':
    binary_label = angle_label_encode([0, -1, -2, -45, -80, -90, 180 / 256. - 180, -180], 180, 180 / 128., mode=0)
    print(binary_label)
    label = angle_label_decode(binary_label, 180, 180 / 128., mode=0)
    print(label)

    # gray_label = angle_label_encode([0, -1, -2, -45, -80, -90, 180/256.-180, -180], 180, 180/128., mode=1)
    # print(gray_label)
    # label = angle_label_decode(gray_label, 180, 180/128., mode=1)
    # print(label)

    # dichotomy_label = angle_label_encode([0, -1, -2, -45, -80, -90, 180 / 256. - 180, -180], 180, 180 / 256., mode=2)
    # print(dichotomy_label)
    # label = angle_label_decode(dichotomy_label, 180, 180 / 256., mode=2)
    # print(label)

    short_dichotomy_label = angle_label_encode([0, -1, -2, -45, -80, -90, 180 / 256. - 180, -180], 180, 180 / 128.,
                                               mode=3)
    print(short_dichotomy_label)
    label = angle_label_decode(short_dichotomy_label, 180, 180 / 128., mode=3)
    print(label)

    # a = angle_label_encode(np.arange(0, 181) * -1, 180, 180/128, mode=0)
    # a = angle_label_decode(a, 180, 180/180, mode=0)
    # print(a)
