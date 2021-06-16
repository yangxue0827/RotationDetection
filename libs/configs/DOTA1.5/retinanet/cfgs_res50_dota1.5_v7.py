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
SAVE_WEIGHTS_INTE = 32000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16

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

VERSION = 'RetinaNet_DOTA1.5_1x_20210609'

"""
RetinaNet-R + 90
FLOPs: 514532602;    Trainable params: 34648686

This is your evaluation result for task 1:

mAP: 0.5649682916896377
ap of each class:
plane:0.7531947980713948,
baseball-diamond:0.7541573137988445,
bridge:0.31858774876726986,
ground-track-field:0.6190645976770403,
small-vehicle:0.3210179615221843,
large-vehicle:0.6927275054950305,
ship:0.79052989541663,
tennis-court:0.8969196357072792,
basketball-court:0.7169922653669782,
storage-tank:0.5883855369085842,
soccer-ball-field:0.43080787784362984,
roundabout:0.6610459264714583,
harbor:0.5016350541688466,
swimming-pool:0.6035774490318886,
helicopter:0.390834028402013,
container-crane:1.5072385129584832e-05

The submitted information is :

Description: RetinaNet_DOTA1.5_1x_20210609_41.6w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

