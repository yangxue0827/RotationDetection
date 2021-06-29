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
LR = 1e-3 * BATCH_SIZE * NUM_GPU
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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
CTR_WEIGHT = 1.0
REG_LOSS_MODE = None

VERSION = 'FCOS_DOTA_1x_20210611'

"""
FCOS
FLOPs: 468484112;    Trainable params: 32090136

This is your result for task 1:

    mAP: 0.6768883021004449
    ap of each class:
    plane:0.872774857044874,
    baseball-diamond:0.7456605755847489,
    bridge:0.45562517475651065,
    ground-track-field:0.5894727115456821,
    small-vehicle:0.7427962860138525,
    large-vehicle:0.6294806602885703,
    ship:0.7589526279294885,
    tennis-court:0.8899116518355858,
    basketball-court:0.7648991572877125,
    storage-tank:0.8169922109873226,
    soccer-ball-field:0.536122347311571,
    roundabout:0.5802443666640051,
    harbor:0.6179626999644399,
    swimming-pool:0.6741887143799705,
    helicopter:0.47824048991233775

The submitted information is :

Description: FCOS_DOTA_1x_20210611_35.1w_
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



