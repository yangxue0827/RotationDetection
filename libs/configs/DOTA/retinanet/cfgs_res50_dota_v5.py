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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = 1  # IoU-Smooth L1

VERSION = 'RetinaNet_DOTA_1x_20201225'

"""
RetinaNet-H + IoU-Smooth L1
FLOPs: 484911740;    Trainable params: 33002916

This is your result for task 1:

mAP: 0.6699231893137383
ap of each class:
plane:0.8833785173522034,
baseball-diamond:0.7627482529936743,
bridge:0.44320593902405797,
ground-track-field:0.6785841556477691,
small-vehicle:0.6303299319074853,
large-vehicle:0.5124927246071527,
ship:0.7277748791449373,
tennis-court:0.8980387801428189,
basketball-court:0.79974279949969,
storage-tank:0.7797862635611005,
soccer-ball-field:0.5409846307060925,
roundabout:0.632179992142947,
harbor:0.5621019025063557,
swimming-pool:0.6734955940136754,
helicopter:0.5240034764561138

The submitted information is :

Description: RetinaNet_DOTA_1x_20201225_45.9w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""


