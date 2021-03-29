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

VERSION = 'RetinaNet_DOTA_2x_20210128'

"""
RSDet-8p 20 epoch
FLOPs: 865678510;    Trainable params: 33148131
This is your result for task 1:

mAP: 0.6720081911437207
ap of each class:
plane:0.884237723275074,
baseball-diamond:0.729444339641071,
bridge:0.43210254062916365,
ground-track-field:0.681519345646089,
small-vehicle:0.7078486578377088,
large-vehicle:0.5469995892993882,
ship:0.7267624293020835,
tennis-court:0.8949116521292252,
basketball-court:0.787517553471782,
storage-tank:0.7972219799903847,
soccer-ball-field:0.5521436820074317,
roundabout:0.6207868886053054,
harbor:0.6099057077868252,
swimming-pool:0.6401352757788455,
helicopter:0.46858550175543046

The submitted information is :

Description: RetinaNet_DOTA_2x_20210128_70.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

"""
