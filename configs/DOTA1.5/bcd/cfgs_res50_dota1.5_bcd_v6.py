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
SAVE_WEIGHTS_INTE = 32000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA1.5_BCD_2x_20210721'

"""
RetinaNet-H + 1-1/(sqrt(bcd)+2)
FLOPs: 485782931;    Trainable params: 33051321
This is your evaluation result for task 1:

    mAP: 0.6078214458191175
    ap of each class:
    plane:0.7960611799772355,
    baseball-diamond:0.7361295784085993,
    bridge:0.4098600873125639,
    ground-track-field:0.6723916268287933,
    small-vehicle:0.4969461358511616,
    large-vehicle:0.6750486907026878,
    ship:0.779737113621211,
    tennis-court:0.8969323316457076,
    basketball-court:0.7407904680051254,
    storage-tank:0.6237031460031418,
    soccer-ball-field:0.4636397916269505,
    roundabout:0.6386039475755304,
    harbor:0.6334257658232346,
    swimming-pool:0.6180235560899322,
    helicopter:0.5246413185159465,
    container-crane:0.01920839511805635

The submitted information is :

Description: RetinaNet_DOTA1.5_BCD_2x_20210721_41.6w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

