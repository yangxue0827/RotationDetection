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
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA_2x_20200915'

"""
v4 + 180

This is your result for task 1:

    mAP: 0.6416730011061602
    ap of each class:
    plane:0.8821742942919781,
    baseball-diamond:0.7440080093044878,
    bridge:0.3831123976957141,
    ground-track-field:0.6589485990451555,
    small-vehicle:0.604798345202343,
    large-vehicle:0.49767737835506365,
    ship:0.6829402579754312,
    tennis-court:0.871338023329556,
    basketball-court:0.7813852482817419,
    storage-tank:0.786012241585395,
    soccer-ball-field:0.521218693379607,
    roundabout:0.6001757439327997,
    harbor:0.5127709102957538,
    swimming-pool:0.5936834817142064,
    helicopter:0.5048513922031709

The submitted information is :

Description: RetinaNet_DOTA_2x_20200915_70.2w
Username: liuqingiqng
Institute: Central South University
Emailadress: liuqingqing@csu.edu.cn
TeamMembers: liuqingqing
"""

