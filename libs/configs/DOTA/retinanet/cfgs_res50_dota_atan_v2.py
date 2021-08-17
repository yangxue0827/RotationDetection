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
SAVE_WEIGHTS_INTE = 27000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

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

VERSION = 'RetinaNet_DOTA_1x_20210725'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta)) + 180, sin^2(theta) + cos^2(theta) = 1
[-90, 90]   sin in [-1, 1]   cos in [0, 1]
FLOPs: 485782956;    Trainable params: 33051321
This is your result for task 1:

    mAP: 0.6528336867191903
    ap of each class:
    plane:0.8837036331487185,
    baseball-diamond:0.744263895056926,
    bridge:0.42742965452955967,
    ground-track-field:0.6736477470629797,
    small-vehicle:0.552582680942137,
    large-vehicle:0.45347433774276424,
    ship:0.674311603759745,
    tennis-court:0.8938471775115743,
    basketball-court:0.8017151903508921,
    storage-tank:0.8009366435350515,
    soccer-ball-field:0.5212471582093509,
    roundabout:0.6367629925980651,
    harbor:0.540825871004929,
    swimming-pool:0.6549176731457441,
    helicopter:0.532839042189418

The submitted information is :

Description: RetinaNet_DOTA_1x_20210725_35.1w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



