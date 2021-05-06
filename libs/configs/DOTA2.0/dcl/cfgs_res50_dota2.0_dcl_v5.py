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
SAVE_WEIGHTS_INTE = 40000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

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

VERSION = 'RetinaNet_DOTA2.0_DCL_B_2x_20210430'

"""
FLOPs: 877875705;    Trainable params: 33486966
This is your evaluation result for task 1:

    mAP: 0.4545546886264481
    ap of each class:
    plane:0.7576838360163581,
    baseball-diamond:0.48071214438681525,
    bridge:0.367934070781644,
    ground-track-field:0.5815236628223699,
    small-vehicle:0.34513208572635395,
    large-vehicle:0.36598807625753177,
    ship:0.46939020389816366,
    tennis-court:0.7545404991770102,
    basketball-court:0.5731304185966594,
    storage-tank:0.5011136945493068,
    soccer-ball-field:0.4053102096300879,
    roundabout:0.49786355787385084,
    harbor:0.35938351489137554,
    swimming-pool:0.5031132619574917,
    helicopter:0.5421417282441151,
    container-crane:0.12772487037593397,
    airport:0.45636926716170634,
    helipad:0.09292929292929293

The submitted information is :

Description: RetinaNet_DOTA2.0_DCL_B_2x_20210430_104w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

