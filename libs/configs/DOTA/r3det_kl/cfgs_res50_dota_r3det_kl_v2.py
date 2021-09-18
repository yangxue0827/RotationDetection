# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2,3'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

KL_TAU = 2.0
KL_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA_R3Det_KL_2x_20210204'

"""
r3det + kl + sqrt tau=2
FLOPs: 1032041748;    Trainable params: 37769656
This is your result for task 1:

mAP: 0.7173336572446897
ap of each class:
plane:0.8868781615672788,
baseball-diamond:0.7471104120398441,
bridge:0.4833580754517835,
ground-track-field:0.6546529008163676,
small-vehicle:0.7508776727371242,
large-vehicle:0.7888314834933955,
ship:0.8652117632464216,
tennis-court:0.8937726660697703,
basketball-court:0.8002827781707408,
storage-tank:0.8208248945650195,
soccer-ball-field:0.567949886040721,
roundabout:0.6151266246875117,
harbor:0.6548357599178879,
swimming-pool:0.6777451703578441,
helicopter:0.5525466095086348

The submitted information is :

Description: RetinaNet_DOTA_R3Det_KL_2x_20210204_102.6w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

