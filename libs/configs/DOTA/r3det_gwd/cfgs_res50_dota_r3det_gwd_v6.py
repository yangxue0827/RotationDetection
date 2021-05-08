# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
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

GWD_TAU = 2.0
GWD_FUNC = tf.sqrt

VERSION = 'RetinaNet_DOTA_R3Det_GWD_2x_20201221'

"""
r3det+gwd (only refine stage) + sqrt tau=2
FLOPs: 1032041709;    Trainable params: 37769656

This is your result for task 1:

mAP: 0.7155604317956082
ap of each class:
plane:0.8817888214941647,
baseball-diamond:0.7752929496489188,
bridge:0.4673063222374159,
ground-track-field:0.6651188544481901,
small-vehicle:0.7584158563046128,
large-vehicle:0.7800322878710797,
ship:0.8671372194827895,
tennis-court:0.8968825767254914,
basketball-court:0.809828240738427,
storage-tank:0.8309499971846301,
soccer-ball-field:0.6051807676755416,
roundabout:0.6112273403144328,
harbor:0.6268850150943335,
swimming-pool:0.6702592464207354,
helicopter:0.48710098129335877

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_2x_20201221_70.2w
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui
"""

