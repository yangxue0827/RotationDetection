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
OMEGA = 180 / 64.
ANGLE_MODE = 0  # {0: BCL, 1: GCL}

VERSION = 'RetinaNet_DOTA1.5_R3Det_DCL_B_2x_20210427'

"""
This is your evaluation result for task 1:

    mAP: 0.6217480545337654
    ap of each class:
    plane:0.7998863281989391,
    baseball-diamond:0.7739785552388819,
    bridge:0.4477637990341936,
    ground-track-field:0.607283305782253,
    small-vehicle:0.5035349771954568,
    large-vehicle:0.7172390525651015,
    ship:0.7863096328213743,
    tennis-court:0.8885274962322568,
    basketball-court:0.7443335028892186,
    storage-tank:0.6697926968777169,
    soccer-ball-field:0.47256405702979887,
    roundabout:0.708824360063318,
    harbor:0.5623745792169526,
    swimming-pool:0.64931968085817,
    helicopter:0.5172269475465152,
    container-crane:0.09900990099009901

The submitted information is :

Description: RetinaNet_DOTA1.5_R3Det_DCL_B_2x_20210427_83.2w
Username: AICyber
Institute: IECAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: Yang Xue; Yang Jirui
"""

