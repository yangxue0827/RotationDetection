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
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA_2x_20190530'

"""
RetinaNet-H + 90
This is your result for task 1:

    mAP: 0.6572506703256068
    ap of each class:
    plane:0.8831119481871824,
    baseball-diamond:0.7554052281871614,
    bridge:0.4217303911789575,
    ground-track-field:0.6707230071220774,
    small-vehicle:0.6592650965532021,
    large-vehicle:0.5111005162900164,
    ship:0.7261407293679227,
    tennis-court:0.9071013790480128,
    basketball-court:0.7822207883168055,
    storage-tank:0.7883844023962553,
    soccer-ball-field:0.544082059014562,
    roundabout:0.6200017658693254,
    harbor:0.5324027345069116,
    swimming-pool:0.6718903394664805,
    helicopter:0.3851996693792289

The submitted information is :

Description: RetinaNet_DOTA_2x_20190530_108w
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue

"""

