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
SAVE_WEIGHTS_INTE = 32000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16

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
USE_IOU_FACTOR = False

VERSION = 'RetinaNet_DOTA1.5_R3Det_2x_20210409'

"""
R3Det
FLOPs: 1120443822;    Trainable params: 41428686
This is your evaluation result for task 1:

    mAP: 0.6291430305933157
    ap of each class:
    plane:0.8021795832205134,
    baseball-diamond:0.7555048021776687,
    bridge:0.4236376011997379,
    ground-track-field:0.6521534153938353,
    small-vehicle:0.502202148805237,
    large-vehicle:0.7048027758323485,
    ship:0.7957512997061391,
    tennis-court:0.8950947111351538,
    basketball-court:0.750823027863591,
    storage-tank:0.6627297716213577,
    soccer-ball-field:0.5411447953750608,
    roundabout:0.6927894787515215,
    harbor:0.5536907761315658,
    swimming-pool:0.66129391387984,
    helicopter:0.573074803983895,
    container-crane:0.09941558441558442

The submitted information is :

Description: RetinaNet_DOTA1.5_R3Det_2x_20210409_108.8w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

"""


