# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def hbox_grid_sample(boxes, point_num_per_line=3):

    bin_w, bin_h = (boxes[:, 2] - boxes[:, 0]) / (point_num_per_line-1), (boxes[:, 3] - boxes[:, 1]) / (point_num_per_line-1)

    shift_x = np.expand_dims(np.arange(0, point_num_per_line), axis=0)
    shift_y = np.expand_dims(np.arange(0, point_num_per_line), axis=0)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift_x = np.reshape(shift_x, [-1, 1])
    shift_y = np.reshape(shift_y, [-1, 1])
    shifts = np.concatenate([shift_x, shift_y], axis=-1)
    shifts = np.reshape(shifts, [-1, point_num_per_line**2*2])
    shifts = np.array(np.tile(shifts, [boxes.shape[0], 1]).reshape([-1, point_num_per_line**2, 2]), np.float32)
    bin_w = np.reshape(bin_w, [-1, 1])
    bin_h = np.reshape(bin_h, [-1, 1])
    shifts[:, :, 0] *= bin_w
    shifts[:, :, 1] *= bin_h
    shifts += boxes[:, np.newaxis, 0:2]
    return shifts.reshape([-1, point_num_per_line**2*2])


def rbox_border_sample(boxes, point_num_per_line=3):
    shift_x = np.arange(0, point_num_per_line-1)
    shift_y = np.arange(0, point_num_per_line-1)
    shift_x = np.reshape(shift_x, [-1, 1])
    shift_y = np.reshape(shift_y, [-1, 1])

    sample_points_list = []
    for i in range(4):
        width = boxes[:, (i*2+2)%8] - boxes[:, (i*2)%8]
        height = boxes[:, (i*2+3)%8] - boxes[:, (i*2+1)%8]
        bin_w, bin_h = width / (point_num_per_line-1), height / (point_num_per_line-1)

        shifts = np.concatenate([shift_x, shift_y], axis=-1)
        shifts = np.array(np.tile(shifts, [boxes.shape[0], 1]).reshape([-1, point_num_per_line-1, 2]), np.float32)
        bin_w = np.reshape(bin_w, [-1, 1])
        bin_h = np.reshape(bin_h, [-1, 1])
        shifts[:, :, 0] *= bin_w
        shifts[:, :, 1] *= bin_h
        shifts += boxes[:, np.newaxis, i*2:(i*2)+2]
        sample_points_list.append(shifts)
    sample_points = np.concatenate(sample_points_list, axis=1)
    return sample_points.reshape([-1, (point_num_per_line - 1) * 4 * 2])


if __name__ == '__main__':
    # print(hbox_grid_sample(np.array([[3, 3, 12, 12], [0, 0, 4, 4]]), 3))
    print(rbox_border_sample(np.array([[3, 3, 12, 3, 12, 12, 3, 12], [0, 0, 4, 0, 4, 4, 0, 4]]), 3))
