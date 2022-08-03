# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0'
LR = 1e-4
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 10000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'ICDAR2015'
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1000
CLASS_NUM = 1

# data augmentation
IMG_ROTATE = True
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

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
VIS_SCORE = 0.72

VERSION = 'RetinaNet_ICDAR2015_R3Det_BCD_2x_20210811'

"""
0.72: Calculated!{"precision": 0.8166501486620417, "recall": 0.7934520943668752, "hmean": 0.8048840048840049, "AP": 0}%
0.7: Calculated!{"precision": 0.8045586808923375, "recall": 0.7987481945113144, "hmean": 0.8016429089151969, "AP": 0}%
0.75: Calculated!{"precision": 0.8252427184466019, "recall": 0.777563793933558, "hmean": 0.8006941001487358, "AP": 0}%
0.65: Calculated!{"precision": 0.7805671780567178, "recall": 0.8083774675012037, "hmean": 0.7942289498580889, "AP": 0}%

0.8048840048840049  0.7829059829059829  0.7472527472527473  0.6857142857142857  0.5860805860805861
0.45421245421245426  0.26568986568986575  0.10891330891330891  0.01807081807081807  0.0009768009768009768

"""

