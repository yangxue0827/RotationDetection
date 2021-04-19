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
SAVE_WEIGHTS_INTE = 32000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16

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
REG_LOSS_MODE = None

# DCL
OMEGA = 180 / 256.  
ANGLE_MODE = 0  # {0: BCL, 1: GCL}

VERSION = 'RetinaNet_DOTA1.5_DCL_B_2x_20210413'

"""
FLOPs: 874390791;    Trainable params: 33390156
This is your evaluation result for task 1:

    mAP: 0.5938264670889287
    ap of each class:
    plane:0.7908842619806756,
    baseball-diamond:0.7270180788541003,
    bridge:0.37851901565797375,
    ground-track-field:0.6267076003692444,
    small-vehicle:0.4690640039292024,
    large-vehicle:0.506520945124799,
    ship:0.732180458044842,
    tennis-court:0.8941057777726436,
    basketball-court:0.7253820677788853,
    storage-tank:0.5962102239817675,
    soccer-ball-field:0.5199832375478927,
    roundabout:0.6881354821535759,
    harbor:0.5236225421727135,
    swimming-pool:0.6556461604666624,
    helicopter:0.5649708903151563,
    container-crane:0.10227272727272728

The submitted information is :

Description: RetinaNet_DOTA1.5_DCL_B_2x_20210413_108.8w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

