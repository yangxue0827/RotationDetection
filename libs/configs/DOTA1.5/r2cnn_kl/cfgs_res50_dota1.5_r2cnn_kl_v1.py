# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 32000
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

# loss
FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0

KL_TAU = 1.0
KL_FUNC = 1   # 0: sqrt  1: log

VERSION = 'FPN_Res50D_DOTA1.5_1x_20210706'

"""
R2CNN + KLD

This is your evaluation result for task 1:

    mAP: 0.6558855975221622
    ap of each class:
    plane:0.8030822937308973,
    baseball-diamond:0.74333164225272,
    bridge:0.47869771243541054,
    round-track-field:0.6013563886785405,
    small-vehicle:0.6597708335329053,
    large-vehicle:0.7311463682097848,
    ship:0.8710069683823015,
    tennis-court:0.8955446605161279,
    basketball-court:0.7537616457403999,
    storage-tank:0.8179552442028525,
    soccer-ball-field:0.4998339693853809,
    roundabout:0.718015276062508,
    harbor:0.6434353553123139,
    swimming-pool:0.7175199752769469,
    helicopter:0.559711226635507,
    container-crane:0.0

The submitted information is :

Description: FPN_Res50D_DOTA1.5_1x_20210706_41.6w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
