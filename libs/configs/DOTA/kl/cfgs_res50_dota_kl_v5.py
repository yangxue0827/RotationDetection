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
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0
REG_LOSS_MODE = 3  # KLD loss

KL_TAU = 2.0
KL_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA_KL_1x_20210201'

"""
RetinaNet-H + kl (fix bug) + sqrt + tau=2
FLOPs: 484911761;    Trainable params: 33002916
This is your result for task 1:

mAP: 0.7128325571761713
ap of each class:
plane:0.884289204525325,
baseball-diamond:0.7653565915743398,
bridge:0.440047662898936,
ground-track-field:0.698238996872059,
small-vehicle:0.7444580421686285,
large-vehicle:0.7248184249702364,
ship:0.843025375274411,
tennis-court:0.8939539261877734,
basketball-court:0.806571747424402,
storage-tank:0.800303800899483,
soccer-ball-field:0.5787146175790521,
roundabout:0.6505316568373755,
harbor:0.6554363687620437,
swimming-pool:0.6686981948609415,
helicopter:0.5380437468075643

The submitted information is :

Description: RetinaNet_DOTA_KL_1x_20210201_45.9w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
