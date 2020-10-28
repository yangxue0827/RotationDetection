# -*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function
import numpy as np
import math
import sys

sys.path.append('../')


def get_code_len(class_range, mode=0):
    if mode in [0, 1, 3]:
        return math.ceil(math.log(class_range, 2))
    elif mode == 2:
        return math.floor(math.log(class_range, 2) + 1) * 2
    else:
        raise Exception('Only support binary, gray and dichotomy coded label')


def get_all_binary_label(num_label, class_range):
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
    :param angle_label: [-90,0) or [-180, 0)
    :param angle_range: 90 or 180
    :param omega:
    :return:
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
    :param angle_label: [-90,0) or [-180, 0)
    :param angle_range: 90 or 180
    :param omega:
    :return:
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


def dichotomy_label_encode(angle_label, angle_range, omega):
    assert (angle_range / omega) % 1 == 0, 'wrong omega'
    # assert -angle_range <= angle_label <= 0, 'wrong angle'

    angle_label = np.array(angle_label, np.int32)
    angle_label = np.divide(angle_label, omega)
    angle_label = np.array(-np.round(angle_label), np.int32)

    angle_range /= omega
    angle_range = int(angle_range)

    inx = angle_label == angle_range
    angle_label[inx] = 0

    dichotomy_label = []
    for al in angle_label:
        # class to list* omega
        nums = [index for index in range(int(angle_range))]

        left = 0
        right = len(nums) - 1
        max_iter = math.floor(math.log(len(nums), 2)) + 1  # maximum number of search
        code = [0 for index in range(int(max_iter) * 2)]
        i = 0

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == al:  # find the num, sign as [1, 1]
                code[i] = 0
                code[i + 1] = 0
                dichotomy_label.append(code)
                break
            elif al < nums[mid]:  # target in the left of nums[mid], sign as [0, 1]
                right = mid - 1
                code[i] = 0
                code[i + 1] = 1
            else:  # target in the right of nums[mid], sign as [1, 0]
                left = mid + 1
                code[i] = 1
                code[i + 1] = 0
            i += 2
    return np.array(dichotomy_label, np.float32)


def dichotomy_label_decode(dichotomy_label, angle_range, omega=1.):
    assert (angle_range / omega) % 1 == 0, 'wrong omega'

    angle_range /= omega
    angle_range = int(angle_range)
    # class to list

    all_angle_label = []

    for dl in dichotomy_label:

        nums = [index for index in range(int(angle_range))]
        left = 0
        right = len(nums) - 1
        i = 0

        while left <= right:
            mid = (left + right) // 2
            if dl[i] == 0 and dl[i + 1] == 0:  # if [1, 1], find the number
                decode_angle_label = np.array(nums[mid], np.float)
                decode_angle_label = angle_range if decode_angle_label == 0 else decode_angle_label
                decode_angle_label = decode_angle_label \
                    if 0 < decode_angle_label <= int(angle_range) \
                    else decode_angle_label - int(angle_range / 2)
                all_angle_label.append(decode_angle_label * omega)
                break
            elif dl[i] == 0 and dl[i + 1] == 1:  # if [0, 1], search in the left
                right = mid - 1
            elif dl[i] == 1 and dl[i + 1] == 0:  # if [1, 0], search in the right
                left = mid + 1
            else:
                all_angle_label.append(0)  # if [0, 0], return 0
                break
            i += 2
    return np.array(all_angle_label, np.float32)


def short_dichotomy_label_encode(angle_label, angle_range, omega):
    assert (angle_range / omega) % 1 == 0, 'wrong omega'

    angle_label = np.array(angle_label, np.int32)
    angle_label = np.divide(angle_label, omega)
    angle_label = np.array(-np.round(angle_label), np.int32)

    angle_range /= omega
    angle_range = int(angle_range)

    inx = angle_label == angle_range
    angle_label[inx] = 0

    nums = [index for index in range(int(angle_range))]
    max_iter = math.floor(math.log(len(nums), 2))  # maximum number of search

    if math.log(len(nums), 2) != float(int(math.log(len(nums), 2))):
        max_iter += 1

    short_dichotomy_label = []
    for al in angle_label:
        left = 0
        right = len(nums) - 1
        code = [0 for index in range(int(max_iter))]
        i = 0
        # if left == right, we find the final target
        # print("angle", al)
        while left <= right:
            if left == right:
                short_dichotomy_label.append(code)
                break
            mid = (left + right) // 2  # left
            left_mid = mid  # right_mid = mid + 1
            if al <= nums[left_mid]:  # if al in the left of mid, sign as 0
                code[i] = 0
                right = left_mid
            else:  # if al in the right of mid, sign as 1
                code[i] = 1
                left = left_mid + 1
            i += 1
            # print("i",i)
            # print("left",left)
            # print("right",right)

    return np.array(short_dichotomy_label, np.float32)


def short_dichotomy_label_decode(dichotomy_label, angle_range, omega=1.):
    assert (angle_range / omega) % 1 == 0, 'wrong omega'

    angle_range /= omega
    angle_range = int(angle_range)

    all_angle_label = []

    for dl in dichotomy_label:

        nums = [index for index in range(int(angle_range))]
        left = 0
        right = len(nums) - 1
        i = 0

        while left <= right:
            if left == right:
                break
            mid = (left + right) // 2  # left
            left_mid = mid  # right_mid = mid + 1
            if dl[i] == 0:  # if [0, 1], search in the left
                right = left_mid
            else:  # if [1, 0], search in the right
                left = left_mid + 1
            i += 1

        decode_angle_label = np.array(nums[left], np.float)  # left == right
        decode_angle_label = angle_range if decode_angle_label == 0 else decode_angle_label
        decode_angle_label = decode_angle_label \
            if 0 < decode_angle_label <= int(angle_range) \
            else decode_angle_label - int(angle_range / 2)
        all_angle_label.append(decode_angle_label * omega)

    return np.array(all_angle_label, np.float32)


def angle_label_encode(angle_label, angle_range, omega=1., mode=0):
    if mode == 0:
        angle_binary_label = binary_label_encode(angle_label, angle_range, omega=omega)
        return angle_binary_label
    elif mode == 1:
        angle_gray_label = gray_label_encode(angle_label, angle_range, omega=omega)
        return angle_gray_label
    elif mode == 2:
        angle_dichotomy_label = dichotomy_label_encode(angle_label, angle_range, omega=omega)
        return angle_dichotomy_label
    elif mode == 3:
        angle_short_dichotomy_label = short_dichotomy_label_encode(angle_label, angle_range, omega)
        return angle_short_dichotomy_label
    else:
        raise Exception('Only support binary, gray and dichotomy coded label')


def angle_label_decode(angle_encode_label, angle_range, omega=1., mode=0):
    if mode == 0:
        angle_label = binary_label_decode(angle_encode_label, angle_range, omega=omega)
    elif mode == 1:
        angle_label = gray_label_decode(angle_encode_label, angle_range, omega=omega)
    elif mode == 2:
        angle_label = dichotomy_label_decode(angle_encode_label, angle_range, omega=omega)
    elif mode == 3:
        angle_label = short_dichotomy_label_decode(angle_encode_label, angle_range, omega=omega)
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
