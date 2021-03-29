# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 40000 * 2
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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA2.0_2x_20210314'

"""
RetinaNet-H + 90
FLOPs: 1054755172;    Trainable params: 33148131
This is your evaluation result for task 1:

mAP: 0.44156413432908415
ap of each class:
plane:0.7402191062015364,
baseball-diamond:0.4645792974915081,
bridge:0.37647950023482696,
ground-track-field:0.5743651999375883,
small-vehicle:0.34562693622352575,
large-vehicle:0.3502289061536839,
ship:0.4695205209241055,
tennis-court:0.7745719743717583,
basketball-court:0.520493288779417,
storage-tank:0.5391885348848654,
soccer-ball-field:0.38803472572261355,
roundabout:0.48309476020235276,
harbor:0.34558533749052234,
swimming-pool:0.5074362374340297,
helicopter:0.4550923397013484,
container-crane:0.10480495563368492,
airport:0.4222649434676708,
helipad:0.08656785306847574

The submitted information is :

Description: RetinaNet_DOTA2.0_2x_20210314_104w (retinanet baseline, https://github.com/yangxue0827/RotationDetection)
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""


