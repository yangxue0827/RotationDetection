# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 0.5

# DCL
OMEGA = 180 / 256.  
ANGLE_MODE = 0  # {0: BCL, 1: GCL}

VERSION = 'RetinaNet_DOTA_DCL_B_2x_20200912'

"""
FLOPs: 872648361;    Trainable params: 33341751
v4 + period loss
This is your result for task 1:

    mAP: 0.673917075709117
    ap of each class:
    plane:0.8891746888406706,
    baseball-diamond:0.7211152849848342,
    bridge:0.41398424290638336,
    ground-track-field:0.6631986431727604,
    small-vehicle:0.6582194691343981,
    large-vehicle:0.5627183585938578,
    ship:0.7379739445312744,
    tennis-court:0.9078627422924039,
    basketball-court:0.7985916610693894,
    storage-tank:0.7902816236238415,
    soccer-ball-field:0.5411469473323773,
    roundabout:0.6024954269211565,
    harbor:0.5430105428794202,
    swimming-pool:0.6786164944083442,
    helicopter:0.6003660649456417

The submitted information is :

Description: RetinaNet_DOTA_DCL_B_2x_20200912_81w
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue
"""

