# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 40000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA2.0_BCD_2x_20210723'

"""
RetinaNet-H + 1-1/(sqrt(bcd)+2)
FLOPs: 487525261;    Trainable params: 33148131

This is your evaluation result for task 1:

    mAP: 0.4748251390678757
    ap of each class:
    plane:0.7694004758275651,
    baseball-diamond:0.4633958753542787,
    bridge:0.39140982919175676,
    ground-track-field:0.5911176240553634,
    small-vehicle:0.41513007763044857,
    large-vehicle:0.4719015966001979,
    ship:0.5700450139195067,
    tennis-court:0.7799866191766394,
    basketball-court:0.567746193623308,
    storage-tank:0.5067637510814136,
    soccer-ball-field:0.36000188921274584,
    roundabout:0.5053053207383681,
    harbor:0.4518474589586364,
    swimming-pool:0.5247063138022362,
    helicopter:0.49441694419726545,
    container-crane:0.12217758007758465,
    airport:0.45779563157266656,
    helipad:0.10370430820178039

The submitted information is :

Description: RetinaNet_DOTA2.0_BCD_2x_20210723_52w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

"""

