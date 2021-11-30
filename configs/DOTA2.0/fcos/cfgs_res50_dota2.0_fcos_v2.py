# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3 * BATCH_SIZE * NUM_GPU
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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
CTR_WEIGHT = 1.0
REG_LOSS_MODE = 0

VERSION = 'FCOS_DOTA2.0_RSDet_1x_20210617'

"""
FCOS + modulated loss
FLOPs: 468608575;    Trainable params: 32097051

This is your evaluation result for task 1:

    mAP: 0.4881184216403336
    ap of each class:
    plane:0.7863183785113876,
    baseball-diamond:0.4707151264649854,
    bridge:0.39470207745499414,
    ground-track-field:0.546908617302247,
    small-vehicle:0.485864491468598,
    large-vehicle:0.46568955587333427,
    ship:0.5621718190273812,
    tennis-court:0.7660422340867759,
    basketball-court:0.5619650407916696,
    storage-tank:0.6488520700802246,
    soccer-ball-field:0.40844718805430663,
    roundabout:0.5262690391113736,
    harbor:0.43593581191541636,
    swimming-pool:0.5588789927215072,
    helicopter:0.4454172078637718,
    container-crane:0.06500235515779557,
    airport:0.4830832997332654,
    helipad:0.17386828390696976

The submitted information is :

Description: FCOS_DOTA2.0_RSDet_1x_20210617_52w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

"""



