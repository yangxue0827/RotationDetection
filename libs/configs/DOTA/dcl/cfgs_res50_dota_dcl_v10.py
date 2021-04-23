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
ANGLE_MODE = 1  # {0: BCL, 1: GCL}

VERSION = 'RetinaNet_DOTA_DCL_G_2x_20201004'

"""
v3 + period loss

This is your result for task 1:

    mAP: 0.6701529258132946
    ap of each class:
    plane:0.8852753615354908,
    baseball-diamond:0.7358487312228768,
    bridge:0.397846525029383,
    ground-track-field:0.643821813715879,
    small-vehicle:0.67198526654463,
    large-vehicle:0.5602121060064053,
    ship:0.740990137738814,
    tennis-court:0.9078254427263777,
    basketball-court:0.7913191496695368,
    storage-tank:0.7794879186223984,
    soccer-ball-field:0.5560141958479111,
    roundabout:0.6190036754323164,
    harbor:0.5381769188233715,
    swimming-pool:0.6618352908439461,
    helicopter:0.5626513534400819

The submitted information is :

Description: RetinaNet_DOTA_DCL_G_2x_20201004_70.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

"""
