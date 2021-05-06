# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
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
ANGLE_RANGE = 180

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 0.5
USE_IOU_FACTOR = True

# DCL
OMEGA = 180 / 256.  
ANGLE_MODE = 0  # {0: BCL, 1: GCL}

VERSION = 'RetinaNet_DOTA2.0_R3Det_DCL_B_2x_20210501'

"""
FLOPs: 1038098744;    Trainable params: 37937921

This is your evaluation result for task 1:

    mAP: 0.48714130776739434
    ap of each class:
    plane:0.7880116046903097,
    baseball-diamond:0.4692647978217154,
    bridge:0.38858081752752943,
    ground-track-field:0.5987126479221437,
    small-vehicle:0.41932698755109493,
    large-vehicle:0.5009640328109685,
    ship:0.5767508546587997,
    tennis-court:0.7559193172453369,
    basketball-court:0.561313088594416,
    storage-tank:0.5843369292867531,
    soccer-ball-field:0.41987150042788446,
    roundabout:0.5008947819556756,
    harbor:0.4142153419620439,
    swimming-pool:0.5404425029378608,
    helicopter:0.5613433762889186,
    container-crane:0.14421137326896488,
    airport:0.4891753744365273,
    helipad:0.05520821042615445

The submitted information is :

Description: RetinaNet_DOTA2.0_R3Det_DCL_B_2x_20210501_104w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

