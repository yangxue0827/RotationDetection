# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
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

GWD_TAU = 2.0
GWD_FUNC = tf.sqrt

VERSION = 'RetinaNet_DOTA2.0_R3Det_GWD_2x_20210425'

"""
r3det+gwd (only refine stage) + sqrt tau=2
FLOPs: 1037517936;    Trainable params: 37921786

This is your evaluation result for task 1:

    mAP: 0.4924996532515483
    ap of each class:
    plane:0.7916743819132743,
    baseball-diamond:0.46747380439844083,
    bridge:0.39221255369800695,
    ground-track-field:0.5777379770806235,
    small-vehicle:0.4324146476590651,
    large-vehicle:0.5321973121549544,
    ship:0.5920285626768007,
    tennis-court:0.7807436606175097,
    basketball-court:0.5750906630386756,
    storage-tank:0.6248682974801154,
    soccer-ball-field:0.43804070336628614,
    roundabout:0.49820542104992493,
    harbor:0.42499687106769646,
    swimming-pool:0.5384411256832793,
    helicopter:0.506867294923433,
    container-crane:0.12692138779095302,
    airport:0.4442338155198331,
    helipad:0.12084527840899882

The submitted information is :

Description: RetinaNet_DOTA2.0_R3Det_GWD_2x_20210425_104w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

