# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.schedules.schedule_1x import *
from configs._base_.datasets.dota_detection import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA_BCD_1x_20210721'

"""
RetinaNet-H + 1-1/(sqrt(bcd)+2)
FLOPs: 484911765;    Trainable params: 33002916
This is your result for task 1:

    mAP: 0.7123346377868429
    ap of each class:
    plane:0.8887915974633945,
    baseball-diamond:0.7580879255573998,
    bridge:0.4515840080232722,
    ground-track-field:0.6646777402763645,
    small-vehicle:0.7404309877818988,
    large-vehicle:0.7219149415987607,
    ship:0.8406801553214951,
    tennis-court:0.9009695293086886,
    basketball-court:0.8126487421623486,
    storage-tank:0.8022797894263551,
    soccer-ball-field:0.5677260871453866,
    roundabout:0.6451663275717179,
    harbor:0.650692992037595,
    swimming-pool:0.6591555368760995,
    helicopter:0.5802132062518686

The submitted information is :

Description: RetinaNet_DOTA_BCD_1x_20210721_35.1w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
