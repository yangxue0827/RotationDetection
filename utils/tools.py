# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import math
import sys
import os
import cv2
import numpy as np
# import tfplot as tfp

from libs.utils.coordinate_convert import forward_convert


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dota_short_names(label):
    DOTA_SHORT_NAMES = {
        'roundabout': 'RA',
        'tennis-court': 'TC',
        'swimming-pool': 'SP',
        'storage-tank': 'ST',
        'soccer-ball-field': 'SBF',
        'small-vehicle': 'SV',
        'ship': 'SH',
        'plane': 'PL',
        'large-vehicle': 'LV',
        'helicopter': 'HC',
        'harbor': 'HA',
        'ground-track-field': 'GTF',
        'bridge': 'BR',
        'basketball-court': 'BC',
        'baseball-diamond': 'BD',
        'container-crane': 'CC',
        'airport': 'AP',
        'helipad': 'HP'
    }

    return DOTA_SHORT_NAMES[label]


def read_dota_gt_and_vis(img, gt_txt):
    txt_data = open(gt_txt, 'r').readlines()
    for i in txt_data:
        if len(i.split(' ')) < 9:
            continue

        gt_box = [int(xy) for xy in i.split(' ')[:8]]
        # gt_label = i.split(' ')[8]
        cv2.line(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color=(0, 0, 255), thickness=3)
        cv2.line(img, (gt_box[2], gt_box[3]), (gt_box[4], gt_box[5]), color=(0, 0, 255), thickness=3)
        cv2.line(img, (gt_box[4], gt_box[5]), (gt_box[6], gt_box[7]), color=(0, 0, 255), thickness=3)
        cv2.line(img, (gt_box[6], gt_box[7]), (gt_box[0], gt_box[1]), color=(0, 0, 255), thickness=3)
    return img


def get_mask(img, boxes):
    boxes = forward_convert(boxes)
    h, w, _ = img.shape
    mask = np.zeros([h, w])
    for b in boxes:
        b = np.reshape(b[0:-1], [4, 2])
        rect = np.array(b, np.int32)
        cv2.fillConvexPoly(mask, rect, 1)
    # mask = cv2.resize(mask, dsize=(h // 16, w // 16))
    mask = np.expand_dims(mask, axis=-1)
    return np.array(mask, np.float32)


# def add_heatmap(feature_maps, name):
#     '''
#     :param feature_maps:[B, H, W, C]
#     :return:
#     '''
#
#     def figure_attention(activation):
#         fig, ax = tfp.subplots()
#         im = ax.imshow(activation, cmap='jet')
#         fig.colorbar(im)
#         return fig
#
#     heatmap = tf.reduce_sum(feature_maps, axis=-1)
#     heatmap = tf.squeeze(heatmap, axis=0)
#     tfp.summary.plot(name, figure_attention, [heatmap])
