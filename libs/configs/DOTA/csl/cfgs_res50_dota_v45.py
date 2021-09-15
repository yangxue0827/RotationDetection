# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 27000 * 2
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
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 2.0
REG_LOSS_MODE = None

# CSL
LABEL_TYPE = 0  # {0: gaussian_label, 1: rectangular_label, 2: pulse_label, 3: triangle_label}
RADIUS = 1
OMEGA = 10

VERSION = 'RetinaNet_DOTA_2x_20200729'

"""
gaussian label, omega=10

This is your result for task 1:

    mAP: 0.6738304898085752
    ap of each class:
    plane:0.8914335719717486,
    baseball-diamond:0.7825554534383872,
    bridge:0.42253905536333874,
    ground-track-field:0.618954473632168,
    small-vehicle:0.682804572312177,
    large-vehicle:0.5450885181789469,
    ship:0.7285446732619127,
    tennis-court:0.9086060935169401,
    basketball-court:0.7934179459801323,
    storage-tank:0.7558745355233758,
    soccer-ball-field:0.5327609959142836,
    roundabout:0.5898646648632354,
    harbor:0.5310054187218879,
    swimming-pool:0.6949448883695547,
    helicopter:0.6290624860805408

The submitted information is :

Description: RetinaNet_DOTA_2x_20200729_75.6w
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue


"""


