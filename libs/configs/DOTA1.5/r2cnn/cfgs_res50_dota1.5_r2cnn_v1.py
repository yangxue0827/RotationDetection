# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
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

VERSION = 'FPN_Res50D_DOTA1.5_1x_20210416'

"""
R2CNN
FLOPs: 1238001921;    Trainable params: 41778832
This is your evaluation result for task 1:

    mAP: 0.6644517842546864
    ap of each class:
    plane:0.8037681771989416,
    baseball-diamond:0.7870306777584335,
    bridge:0.4794652822872299,
    ground-track-field:0.6255039590117601,
    small-vehicle:0.6560384515873214,
    large-vehicle:0.7133310232113534,
    ship:0.8634741652348884,
    tennis-court:0.8974358406150273,
    basketball-court:0.7628652235221789,
    storage-tank:0.7622601634906873,
    soccer-ball-field:0.4973441797328644,
    roundabout:0.6754774857555178,
    harbor:0.63474384391541,
    swimming-pool:0.7314676462716916,
    helicopter:0.5844230118555245,
    container-crane:0.1565994166261546

The submitted information is :

Description: FPN_Res50D_DOTA1.5_1x_20210416_54.4w
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: yangxue
"""
