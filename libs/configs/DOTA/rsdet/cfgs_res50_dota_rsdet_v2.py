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
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_EPOCH = [18, 24, 30]
MAX_EPOCH = 25
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

VERSION = 'RetinaNet_DOTA_2x_20201128'

"""
RSDet-8p

This is your result for task 1:

mAP: 0.6727423650267537
ap of each class:
plane:0.8839346472596076,
baseball-diamond:0.7104926703230673,
bridge:0.4330823738329618,
ground-track-field:0.6508970563363848,
small-vehicle:0.6849253155621244,
large-vehicle:0.6102196316871491,
ship:0.7961936701620749,
tennis-court:0.8947516994949227,
basketball-court:0.7455840121634438,
storage-tank:0.7672332259150044,
soccer-ball-field:0.5496680505639349,
roundabout:0.6350320699137543,
harbor:0.5842793618604702,
swimming-pool:0.6505404500942658,
helicopter:0.4943012402321392

The submitted information is :

Description: RetinaNet_DOTA_2x_20201128_162w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

"""


