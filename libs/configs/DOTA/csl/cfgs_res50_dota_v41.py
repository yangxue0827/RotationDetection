# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1"
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
LABEL_TYPE = 2  # {0: gaussian_label, 1: rectangular_label, 2: pulse_label, 3: triangle_label}
RADIUS = 4
OMEGA = 1

VERSION = 'RetinaNet_DOTA_2x_20200724'

"""
pulse label, omega=1

This is your result for task 1:

    mAP: 0.687302111248375
    ap of each class:
    plane:0.8909342530317464,
    baseball-diamond:0.7922712750998728,
    bridge:0.45785959546550553,
    ground-track-field:0.6794415673568659,
    small-vehicle:0.6658176543655151,
    large-vehicle:0.5628798425523656,
    ship:0.7155442805781995,
    tennis-court:0.9080095714731273,
    basketball-court:0.8050826610416045,
    storage-tank:0.766114706284049,
    soccer-ball-field:0.5852681707570853,
    roundabout:0.6100481551283217,
    harbor:0.5992335408254846,
    swimming-pool:0.6950960106316608,
    helicopter:0.5759303841342222

The submitted information is :

Description: RetinaNet_DOTA_2x_20200724_70.2w_
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

"""


