# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0'
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 10000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'ICDAR2015'
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1000
CLASS_NUM = 1

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

# post-processing
VIS_SCORE = 0.92

VERSION = 'RetinaNet_ICDAR2015_R3Det_BCD_1x_20210811'

"""
0.92: Calculated!{"precision": 0.8148337595907928, "recall": 0.7669715936446798, "hmean": 0.7901785714285715, "AP": 0}%
0。93: Calculated!{"precision": 0.8222106360792493, "recall": 0.7592681752527685, "hmean": 0.7894868585732167, "AP": 0}%
0.94: Calculated!{"precision": 0.8304904051172708, "recall": 0.7501203659123736, "hmean": 0.7882620794333418, "AP": 0}%
0.9: Calculated!{"precision": 0.8011928429423459, "recall": 0.7761194029850746, "hmean": 0.7884568354120812, "AP": 0}%

0.7901785714285715  0.7658730158730158  0.7281746031746031  0.679563492063492  0.5952380952380953
0.4568452380952381  0.28174603174603174  0.10416666666666666  0.020337301587301588  0.

"""
