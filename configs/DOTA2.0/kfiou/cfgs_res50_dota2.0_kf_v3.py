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
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_DOTA2.0_KF_1x_20210904'

"""
RetinaNet-H + kfiou (exp(1-IoU)-1)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

FLOPs: 487527642;    Trainable params: 33148131

This is your evaluation result for task 1:

    mAP: 0.48036732934581633
    ap of each class:
    plane:0.7797565606743677,
    baseball-diamond:0.4881589892507902,
    bridge:0.394693953354765,
    ground-track-field:0.6011431027821553,
    small-vehicle:0.4060010176839843,
    large-vehicle:0.4425839378634837,
    ship:0.5442586582492351,
    tennis-court:0.7975883024719737,
    basketball-court:0.5597256955856627,
    storage-tank:0.5257633480872891,
    soccer-ball-field:0.40634618540482653,
    roundabout:0.5102452174317035,
    harbor:0.44655732559476546,
    swimming-pool:0.544513605812594,
    helicopter:0.48219742508221414,
    container-crane:0.11009217070573671,
    airport:0.5151000535310445,
    helipad:0.09188637865810127

The submitted information is :

Description: RetinaNet_DOTA2.0_KF_1x_20210904_52w_cpunms
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



