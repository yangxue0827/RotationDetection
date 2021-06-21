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

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
METHOD = 'R'
ANCHOR_RATIOS = [1, 1 / 3., 3.]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA_1x_20210610'

"""
RetinaNet-R + 90
FLOPs: 512292465;    Trainable params: 34524216
This is your result for task 1:

    mAP: 0.6725084704054188
    ap of each class:
    plane:0.8928344652701296,
    baseball-diamond:0.7304554873104306,
    bridge:0.34861749965443384,
    ground-track-field:0.6454757711323661,
    small-vehicle:0.7358328451641203,
    large-vehicle:0.7333202044070867,
    ship:0.829537451899966,
    tennis-court:0.8991725822927377,
    basketball-court:0.7414258616115545,
    storage-tank:0.7907933807503915,
    soccer-ball-field:0.5718970329039421,
    roundabout:0.5956995083477648,
    harbor:0.51027563083865,
    swimming-pool:0.6614448075148596,
    helicopter:0.4008445269828484

The submitted information is :

Description: RetinaNet_DOTA_1x_20210610_35.1w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

