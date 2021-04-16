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
SAVE_WEIGHTS_INTE = 40000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

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

VERSION = 'RetinaNet_DOTA2.0_R3Det_2x_20210410'

"""
R3Det
FLOPs: 1124094684;    Trainable params: 41530106
This is your evaluation result for task 1:

    mAP: 0.48433016270779095
    ap of each class:
    plane:0.7920200738879564,
    baseball-diamond:0.4551905836447281,
    bridge:0.38770671301480925,
    ground-track-field:0.5746292874983392,
    small-vehicle:0.4233527286939054,
    large-vehicle:0.4946097693467611,
    ship:0.5719757986261312,
    tennis-court:0.7746529016429764,
    basketball-court:0.5823408845309606,
    storage-tank:0.5705764020949572,
    soccer-ball-field:0.4358035856805696,
    roundabout:0.5031340218974659,
    harbor:0.3923683731983316,
    swimming-pool:0.5448657101168235,
    helicopter:0.5139043646487352,
    container-crane:0.11816647583920607,
    airport:0.4134305115134352,
    helipad:0.16921474286414373

The submitted information is :

Description: RetinaNet_DOTA2.0_R3Det_2x_20210410_136w
Username: DetectionTeamCSU
Institute: UCAS
Emailadress: yangxue@csu.edu.cn
TeamMembers: yangxue
"""


