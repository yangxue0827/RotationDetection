5# -*- coding: utf-8 -*-
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

# bbox head
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA_2x_20210727'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta)) + 180, sin^2(theta) + cos^2(theta) = 1
[-90, 90]   sin in [-1, 1]   cos in [0, 1]
FLOPs: 862193708;    Trainable params: 33051321

This is your result for task 1:

    mAP: 0.6578072042544206
    ap of each class:
    plane:0.8876928590825857,
    baseball-diamond:0.7080134003984874,
    bridge:0.41518046969184713,
    ground-track-field:0.6641334517905239,
    small-vehicle:0.6393647386261688,
    large-vehicle:0.44946718683582615,
    ship:0.7117757018402356,
    tennis-court:0.9014456759798808,
    basketball-court:0.7988175573739552,
    storage-tank:0.7810617259611029,
    soccer-ball-field:0.5544134674919993,
    roundabout:0.6054450098129257,
    harbor:0.5322431963201936,
    swimming-pool:0.6650098753335547,
    helicopter:0.5530437472770227

The submitted information is :

Description: RetinaNet_DOTA_2x_20210727_70.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



