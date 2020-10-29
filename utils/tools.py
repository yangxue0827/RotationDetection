# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import math
import sys
import os
import cv2


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
        'baseball-diamond': 'BD'
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
