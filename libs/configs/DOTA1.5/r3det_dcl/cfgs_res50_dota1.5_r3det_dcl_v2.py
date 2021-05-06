# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2,3'
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
OMEGA = 180 / 256.  
ANGLE_MODE = 0  # {0: BCL, 1: GCL}

VERSION = 'RetinaNet_DOTA1.5_R3Det_DCL_B_2x_20210430'

"""
FLOPs: 1265719914;    Trainable params: 37836501
This is your evaluation result for task 1:

    mAP: 0.6198407538220302
    ap of each class:
    plane:0.8033864356092784,
    baseball-diamond:0.7439037352653705,
    bridge:0.4421148358813759,
    ground-track-field:0.620146228768308,
    small-vehicle:0.5024054473756266,
    large-vehicle:0.721512585670462,
    ship:0.788606715518134,
    tennis-court:0.8928454376156971,
    basketball-court:0.740436245979083,
    storage-tank:0.6704596019487904,
    soccer-ball-field:0.45461722530887605,
    roundabout:0.6906308439520609,
    harbor:0.5616533098425578,
    swimming-pool:0.6405632905887441,
    helicopter:0.5514057433494514,
    container-crane:0.0927643784786642

The submitted information is :

Description: RetinaNet_DOTA1.5_R3Det_DCL_B_2x_20210430_83.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

