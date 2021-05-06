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

# bbox head
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA2.0_2x_20210502'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta))
FLOPs: 1056933262;    Trainable params: 33196536

This is your evaluation result for task 1:

    mAP: 0.440500285618368
    ap of each class:
    plane:0.7298559412973734,
    baseball-diamond:0.47606424522583657,
    bridge:0.3455064545545123,
    ground-track-field:0.575936688491863,
    small-vehicle:0.34377105197328467,
    large-vehicle:0.3471444853526944,
    ship:0.4438882675504161,
    tennis-court:0.7653275467525217,
    basketball-court:0.5413984534981682,
    storage-tank:0.4985469386722329,
    soccer-ball-field:0.4074311865739111,
    roundabout:0.4951311257604898,
    harbor:0.35180428080527854,
    swimming-pool:0.49591835025994685,
    helicopter:0.5087884081045285,
    container-crane:0.05785811526244139,
    airport:0.4486014855029544,
    helipad:0.09603211549217132

The submitted information is :

Description: RetinaNet_DOTA2.0_2x_20210502_104w
Username: DetectionTeamCSU
Institute: UCAS
Emailadress: yangxue@csu.edu.cn
TeamMembers: yangxue
"""



