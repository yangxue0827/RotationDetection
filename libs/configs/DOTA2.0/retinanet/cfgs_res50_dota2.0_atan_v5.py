# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
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

# bbox head
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA2.0_2x_20210813'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta)) + 180, sin^2(theta) + cos^2(theta) = 1
[-90, 90]   sin in [-1, 1]   cos in [0, 1]

This is your evaluation result for task 1:

    mAP: 0.43915134481218604
    ap of each class:
    plane:0.7452025314053422,
    baseball-diamond:0.465345437959971,
    bridge:0.37478072189595657,
    ground-track-field:0.6131133292307437,
    small-vehicle:0.3413886559945709,
    large-vehicle:0.31453542492957676,
    ship:0.4381808084156131,
    tennis-court:0.759258101199081,
    basketball-court:0.5461806480060146,
    storage-tank:0.5176508913492636,
    soccer-ball-field:0.3874204508627029,
    roundabout:0.48870818927345167,
    harbor:0.37433083038974285,
    swimming-pool:0.5346951854402754,
    helicopter:0.47274904931609435,
    container-crane:0.09784493435719593,
    airport:0.37623883629166993,
    helipad:0.0571001803020811

The submitted information is :

Description: RetinaNet_DOTA2.0_2x_20210813_52w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



