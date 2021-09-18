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
SAVE_WEIGHTS_INTE = 40000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1.0 / 8.0
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0
REG_LOSS_MODE = 3  # KLD loss

KL_TAU = 2.0
KL_FUNC = 1   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA2.0_KL_1x_20210316'

"""
RetinaNet-H + kl + log + tau=2
FLOPs: 487525264;    Trainable params: 33148131

This is your evaluation result for task 1:

mAP: 0.47693258859717275
ap of each class:
plane:0.7751585349971137,
baseball-diamond:0.47365856108350096,
bridge:0.3986354628355674,
ground-track-field:0.5845588244150076,
small-vehicle:0.4177122395617935,
large-vehicle:0.47090121140047525,
ship:0.5840082300122357,
tennis-court:0.778401423346234,
basketball-court:0.5707893690066328,
storage-tank:0.5095359115756787,
soccer-ball-field:0.3930725302943191,
roundabout:0.5022412902162444,
harbor:0.4550802207119013,
swimming-pool:0.5328073428939191,
helicopter:0.46242670615253884,
container-crane:0.08702004541434318,
airport:0.48050945109297416,
helipad:0.10826923973863053

The submitted information is :

Description: RetinaNet_DOTA2.0_KL_1x_20210316_68w
Username: DetectionTeamCSU
Institute: UCAS
Emailadress: yangxue@csu.edu.cn
TeamMembers: yangxue
"""
