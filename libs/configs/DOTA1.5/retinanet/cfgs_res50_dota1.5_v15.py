# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 32000 * 2
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

VERSION = 'RetinaNet_DOTA1.5_2x_20210503'

"""
retinanet-180
FLOPs: 862193566;    Trainable params: 33051321
This is your evaluation result for task 1:

    mAP: 0.5610401956363628
    ap of each class:
    plane:0.7875460889719075,
    baseball-diamond:0.7387149439907101,
    bridge:0.37805634835169255,
    ground-track-field:0.6027540626851335,
    small-vehicle:0.39196873401154814,
    large-vehicle:0.44856173941511396,
    ship:0.6361687880707679,
    tennis-court:0.8540536580109865,
    basketball-court:0.7361866563787661,
    storage-tank:0.5859551738959604,
    soccer-ball-field:0.4861815416240239,
    roundabout:0.6460535463270851,
    harbor:0.4854161897067069,
    swimming-pool:0.6303611675387102,
    helicopter:0.4669898500543662,
    container-crane:0.10167464114832536

The submitted information is :

Description: RetinaNet_DOTA1.5_2x_20210503_83.2w
Username: AICyber
Institute: IECAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: Yang Xue; Yang Jirui
"""

