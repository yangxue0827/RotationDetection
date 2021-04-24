# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1"
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
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0

VERSION = 'RetinaNet_DOTA2.0_RSDet_2x_20210420'

"""
RSDet-8p
FLOPs: 680522298;    Trainable params: 33293346

This is your evaluation result for task 1:

    mAP: 0.467140470448159
    ap of each class:
    plane:0.7726223340937479,
    baseball-diamond:0.47743060672028564,
    bridge:0.3911535390606563,
    ground-track-field:0.5944893217136774,
    small-vehicle:0.3983351425006348,
    large-vehicle:0.39310037449027385,
    ship:0.49591155195452585,
    tennis-court:0.7648821678895156,
    basketball-court:0.5618059409483068,
    storage-tank:0.5243790498027646,
    soccer-ball-field:0.41333134879331546,
    roundabout:0.5145590119591201,
    harbor:0.44300878977240044,
    swimming-pool:0.5279975364577072,
    helicopter:0.45158822358142103,
    container-crane:0.09442517964997484,
    airport:0.46860079031041624,
    helipad:0.12090755836811845

The submitted information is :

Description: RetinaNet_DOTA2.0_RSDet_2x_20210420_136w
Username: DetectionTeamCSU
Institute: UCAS
Emailadress: yangxue@csu.edu.cn
TeamMembers: yangxue

"""


