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
REG_LOSS_MODE = None

VERSION = 'FCOS_DOTA1.5_1x_20210616'

"""
FCOS
FLOPs: 468525597;    Trainable params: 32092441

This is your evaluation result for task 1:

    mAP: 0.6104894338514848
    ap of each class:
    plane:0.7866671512159925,
    baseball-diamond:0.7250219297435628,
    bridge:0.44305003837464796,
    ground-track-field:0.5956968831530797,
    small-vehicle:0.5624770890218149,
    large-vehicle:0.6402938804199785,
    ship:0.7805890817340941,
    tennis-court:0.8940127342072589,
    basketball-court:0.7145081047629066,
    storage-tank:0.7332266671689682,
    soccer-ball-field:0.49508360987791955,
    roundabout:0.6647400680428448,
    harbor:0.5577985168682843,
    swimming-pool:0.6326260452275098,
    helicopter:0.4476034051277771,
    container-crane:0.094435736677116

The submitted information is :

Description: FCOS_DOTA1.5_1x_20210616_41.6w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



