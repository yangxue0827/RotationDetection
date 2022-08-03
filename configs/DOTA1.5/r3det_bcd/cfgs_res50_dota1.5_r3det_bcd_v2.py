# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 32000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16

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
BCD_FUNC = 1   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA1.5_R3Det_BCD_2x_20210810'

"""
r3det + bcd + sqrt tau=2
FLOPs: 1033867209;    Trainable params: 37820366
This is your evaluation result for task 1:

    mAP: 0.6353263818844566
    ap of each class:
    plane:0.8021548950247576,
    baseball-diamond:0.7376028696443753,
    bridge:0.397348527723505,
    ground-track-field:0.6506021951409509,
    small-vehicle:0.5689939469590451,
    large-vehicle:0.7450276705989696,
    ship:0.8668145831753968,
    tennis-court:0.8967502935864855,
    basketball-court:0.7518688601692555,
    storage-tank:0.6639095971376509,
    soccer-ball-field:0.48755925682141943,
    roundabout:0.6472729348972853,
    harbor:0.6425737042855099,
    swimming-pool:0.643592589145954,
    helicopter:0.5249411726317311,
    container-crane:0.13820901320901322

The submitted information is :

Description: RetinaNet_DOTA1.5_R3Det_BCD_2x_20210810_83.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

