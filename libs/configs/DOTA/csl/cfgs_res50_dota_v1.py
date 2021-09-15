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
ANGLE_WEIGHT = 0.5
REG_LOSS_MODE = None

# CSL
LABEL_TYPE = 0  # {0: gaussian_label, 1: rectangular_label, 2: pulse_label, 3: triangle_label}
RADIUS = 6
OMEGA = 1

VERSION = 'RetinaNet_DOTA_CSL_2x_20200912'

"""
gaussian label, omega=1, r=6

This is your result for task 1:

    mAP: 0.6569045253137319
    ap of each class:
    plane:0.8867964561407189,
    baseball-diamond:0.7621447630919623,
    bridge:0.4055336223125793,
    ground-track-field:0.6106504178923284,
    small-vehicle:0.6676809751876145,
    large-vehicle:0.5149858260302362,
    ship:0.7359685211242706,
    tennis-court:0.9083268053106571,
    basketball-court:0.7900280506333788,
    storage-tank:0.7926176857759603,
    soccer-ball-field:0.5518343006559859,
    roundabout:0.5743420301956235,
    harbor:0.537573063205451,
    swimming-pool:0.6545719624624141,
    helicopter:0.4605133996867981

The submitted information is :

Description: RetinaNet_DOTA_CSL_2x_20200912_70.2w
Username: liuqingiqng
Institute: Central South University
Emailadress: liuqingqing@csu.edu.cn
TeamMembers: liuqingqing

"""


