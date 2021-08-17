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
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA1.5_1x_20210813'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta)) + 180, sin^2(theta) + cos^2(theta) = 1
[-90, 90]   sin in [-1, 1]   cos in [0, 1]

This is your evaluation result for task 1:

    mAP: 0.571662845101055
    ap of each class:
    plane:0.785860352359881,
    baseball-diamond:0.7487482192331396,
    bridge:0.3855323501874327,
    ground-track-field:0.647768350234277,
    small-vehicle:0.40264427732383384,
    large-vehicle:0.4520002488728411,
    ship:0.6590035655147902,
    tennis-court:0.8961089944878728,
    basketball-court:0.7273552261544072,
    storage-tank:0.6118851021052684,
    soccer-ball-field:0.499722184305952,
    roundabout:0.6747025379899058,
    harbor:0.5287795104707887,
    swimming-pool:0.6348622406841741,
    helicopter:0.4609098155106093,
    container-crane:0.03072254618170631

The submitted information is :

Description: RetinaNet_DOTA1.5_1x_20210813_41.6w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



