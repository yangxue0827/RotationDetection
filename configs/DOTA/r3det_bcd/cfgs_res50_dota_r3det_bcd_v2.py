# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
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

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA_R3Det_BCD_2x_20210810'

"""
r3det + bcd
FLOPs: 1032041778;   Trainable params: 37769656

This is your result for task 1:

    mAP: 0.7222480357984923
    ap of each class:
    plane:0.8898756824231284,
    baseball-diamond:0.7676544745174824,
    bridge:0.47795129029262956,
    ground-track-field:0.663397739245387,
    small-vehicle:0.7622637231241527,
    large-vehicle:0.7920074475173311,
    ship:0.8689683384597622,
    tennis-court:0.8981123558740035,
    basketball-court:0.8004899297929988,
    storage-tank:0.8306977181036257,
    soccer-ball-field:0.601055761432403,
    roundabout:0.612028212429282,
    harbor:0.6524181685640427,
    swimming-pool:0.6734202558055511,
    helicopter:0.5433794393956056

The submitted information is :

Description: RetinaNet_DOTA_R3Det_BCD_2x_20210810_70.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue


This is your result for task 1:

    mAP : 0.7280794733915879
    ap of each class :
    plane:0.8932453918456619,
    baseball-diamond:0.7735528462449153,
    bridge:0.47832413472539115,
    ground-track-field:0.6652754312793525,
    small-vehicle:0.7679064586789398,
    large-vehicle:0.8013642038470755,
    ship:0.8739410214974208,
    tennis-court:0.9081478484603636,
    basketball-court:0.8290351304685135,
    storage-tank:0.8340053848928062,
    soccer-ball-field:0.601594478455151,
    roundabout:0.6149752753172619,
    harbor:0.6578050522157939,
    swimming-pool:0.6786535176826928,
    helicopter:0.543365925262477

The submitted information is :

Description: RetinaNet_DOTA_R3Det_BCD_2x_20210810_70.2w_cpunms
Username: WWT-SJTU
Institute: SJTU
Emailadress: wwt117@sjtu.edu.cn
TeamMembers: yangxue, wangwentao
"""

