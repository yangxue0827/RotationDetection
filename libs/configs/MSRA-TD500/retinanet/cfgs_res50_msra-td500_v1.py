# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 5000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'MSRA-TD500'
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
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
REG_LOSS_MODE = None

# post-processing
VIS_SCORE = 0.93

VERSION = 'RetinaNet_MSRA_TD500_2x_20210223'

"""
FLOPs: 472715448;    Trainable params: 32325246
0.93: Calculated!{"precision": 0.7953091684434968, "recall": 0.6408934707903781, "hmean": 0.7098001902949571, "AP": 0}%
0.90: Calculated!{"precision": 0.779835390946502, "recall": 0.6512027491408935, "hmean": 0.7097378277153558, "AP": 0}%
0.85: Calculated!{"precision": 0.7549019607843137, "recall": 0.6615120274914089, "hmean": 0.7051282051282052, "AP": 0}%
0.95: Calculated!{"precision": 0.8040089086859689, "recall": 0.6202749140893471, "hmean": 0.700290979631426, "AP": 0}%
0.80: Calculated!{"precision": 0.7358490566037735, "recall": 0.6701030927835051, "hmean": 0.7014388489208633, "AP": 0}%
0.75: Calculated!{"precision": 0.7197802197802198, "recall": 0.6752577319587629, "hmean": 0.696808510638298, "AP": 0}%

Hmean50:95 = 0.7098001902949571 0.6717411988582301 0.6241674595623216 0.5632730732635586 0.47002854424357754
             0.36726926736441484 0.22835394862036157 0.12559467174119887 0.024738344433872503 0.003805899143672693
0.37887725975261655
"""


