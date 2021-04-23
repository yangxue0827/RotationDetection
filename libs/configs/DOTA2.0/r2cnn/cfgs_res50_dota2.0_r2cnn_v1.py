# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 40000
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

VERSION = 'FPN_Res50D_DOTA2.0_1x_20210419'

"""
R2CNN
FLOPs: 1238358557;    Trainable params: 41791132

This is your evaluation result for task 1:

    mAP: 0.5234759167701059
    ap of each class:
    plane:0.7980263807994686,
    baseball-diamond:0.5040420743072365,
    bridge:0.4054416907024359,
    ground-track-field:0.6211356797664414,
    small-vehicle:0.585125118187571,
    large-vehicle:0.5215541280742031,
    ship:0.666279086558348,
    tennis-court:0.7773580788348857,
    basketball-court:0.5859180577370384,
    storage-tank:0.7340726762524007,
    soccer-ball-field:0.42389526398486815,
    roundabout:0.5512795363468848,
    harbor:0.44366789061959877,
    swimming-pool:0.5732843312238601,
    helicopter:0.5230771307504916,
    container-crane:0.16472019343420496,
    airport:0.39625862292530317,
    helipad:0.14743056135666283

The submitted information is :

Description: FPN_Res50D_DOTA2.0_1x_20210419_68w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
