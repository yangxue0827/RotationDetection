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
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

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

VERSION = 'RetinaNet_DOTA_R3Det_DCL_B_2x_20201024'

"""
FLOPs: 1263438187;    Trainable params: 37785791
This is your result for task 1:

    mAP: 0.7121184762618237
    ap of each class:
    plane:0.8879877475181992,
    baseball-diamond:0.778076165652175,
    bridge:0.46840197829363145,
    ground-track-field:0.6584321304370467,
    small-vehicle:0.7486973752001116,
    large-vehicle:0.7496100108721503,
    ship:0.8569940719256488,
    tennis-court:0.9023482027992554,
    basketball-court:0.7931574284009301,
    storage-tank:0.8405654367548424,
    soccer-ball-field:0.5659071452082054,
    roundabout:0.6376705402201622,
    harbor:0.577215213506967,
    swimming-pool:0.6762310908612682,
    helicopter:0.5404826062767616

The submitted information is :

Description: RetinaNet_DOTA_R3Det_DCL_B_2x_20201024_97.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

