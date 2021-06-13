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
SAVE_WEIGHTS_INTE = 40000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
METHOD = 'R'
ANCHOR_RATIOS = [1, 1 / 3., 3.]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA2.0_1x_20210609'

"""
RetinaNet-R + 90
FLOPs: 519012858;    Trainable params: 34897626

This is your evaluation result for task 1:

mAP: 0.42043734008442946
ap of each class:
plane:0.7094945917525478,
baseball-diamond:0.4760893069822692,
bridge:0.3173955065820555,
ground-track-field:0.5960202559941057,
small-vehicle:0.31675346911026486,
large-vehicle:0.44496878068913814,
ship:0.4456975630606699,
tennis-court:0.7714685791376327,
basketball-court:0.5259253641688694,
storage-tank:0.4964239135307835,
soccer-ball-field:0.3420982282408876,
roundabout:0.49416930025396094,
harbor:0.33584285326294183,
swimming-pool:0.46714782477759786,
helicopter:0.3332557070917558,
container-crane:0.00022132462789796936,
airport:0.3850132025324559,
helipad:0.1098863497238967

The submitted information is :

Description: RetinaNet_DOTA2.0_2x_20210609_52w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

