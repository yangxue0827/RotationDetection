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
LR = 1e-3 * BATCH_SIZE * NUM_GPU
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
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
CTR_WEIGHT = 1.0
REG_LOSS_MODE = 0

VERSION = 'FCOS_DOTA1.5_RSDet_1x_20210617'

"""
FCOS + modulated loss
FLOPs: 468525597;    Trainable params: 32092441

This is your evaluation result for task 1:

    mAP: 0.621811745576584
    ap of each class:
    plane:0.7938999239367199,
    baseball-diamond:0.740065627981387,
    bridge:0.4496195731750868,
    ground-track-field:0.6093950686287898,
    small-vehicle:0.5643574518958718,
    large-vehicle:0.6058184462449729,
    ship:0.7801808948640423,
    tennis-court:0.8946595109705113,
    basketball-court:0.7087307071702325,
    storage-tank:0.735681567575362,
    soccer-ball-field:0.5121123427253175,
    roundabout:0.6833713876024533,
    harbor:0.5623638432683176,
    swimming-pool:0.6829058209388849,
    helicopter:0.5228573021360805,
    container-crane:0.10296846011131726

The submitted information is :

Description: FCOS_DOTA1.5_RSDet_1x_20210617_41.6w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



