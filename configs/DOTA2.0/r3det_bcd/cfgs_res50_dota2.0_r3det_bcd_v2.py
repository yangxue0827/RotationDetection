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
SAVE_WEIGHTS_INTE = 40000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

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

VERSION = 'RetinaNet_DOTA2.0_R3Det_BCD_2x_20210810'

"""
r3det + bcd
FLOPs: 1037518071;    Trainable params: 37921786

This is your evaluation result for task 1:

    mAP: 0.497138597688435
    ap of each class:
    plane:0.7907544633930063,
    baseball-diamond:0.4394392922362394,
    bridge:0.3763375229802902,
    ground-track-field:0.5997888100954012,
    small-vehicle:0.43404846038928885,
    large-vehicle:0.536744163312537,
    ship:0.6055920751310522,
    tennis-court:0.7766541023795874,
    basketball-court:0.582857433858447,
    storage-tank:0.5778112278875461,
    soccer-ball-field:0.4450066921350499,
    roundabout:0.5013528371851708,
    harbor:0.45911705094299177,
    swimming-pool:0.5564854560398057,
    helicopter:0.5055087958993411,
    container-crane:0.21503163546446838,
    airport:0.41592776348258886,
    helipad:0.13003697557901722

The submitted information is :

Description: RetinaNet_DOTA2.0_R3Det_BCD_2x_20210810_104w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

