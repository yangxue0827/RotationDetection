# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 2000 * 2
DECAY_EPOCH = [8, 11, 20]
MAX_EPOCH = 12
WARM_EPOCH = 1 / 16.
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'FDDB'
CLASS_NUM = 1

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_SUBNET_CONV = 4
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [1, 1 / 1.5, 1.5]
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 0.5

# DCL
OMEGA = 180 / 64.
ANGLE_MODE = 0  # {0: BCL, 1: GCL}

# eval
USE_07_METRIC = False

VERSION = 'RetinaNet_FDDB_DCL_B_2x_20211107'

"""
2007
cls : face|| Recall: 0.9696969696969697 || Precison: 0.5510763209393347|| AP: 0.9062908374246792
F1:0.9435920247922595 P:0.9729334308705194 R:0.9159779614325069
mAP is : 0.9062908374246792

2012
cls : face|| Recall: 0.9696969696969697 || Precison: 0.5512920908379013|| AP: 0.9615628302927102
F1:0.9435920247922595 P:0.9729334308705194 R:0.9159779614325069
mAP is : 0.9615628302927102

AP50:95
0.9615628302927102  0.9490404295419013  0.9189179490693047  0.893624581242626  0.8521736469223554
0.7406406703627281  0.503127785877697  0.2260002904357899  0.031683276147656605  0.00041642957221458226

"""

