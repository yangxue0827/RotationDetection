# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 27000
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
FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0

KL_TAU = 1.0
KL_FUNC = 1   # 0: sqrt  1: log

VERSION = 'FPN_Res50D_DOTA_KL_1x_20210704'

"""
R2CNN + KLD
FLOPs: 1024153317;    Trainable params: 41772682

This is your result for task 1:

    mAP: 0.721582549954958
    ap of each class:
    plane:0.8849648943434543,
    baseball-diamond:0.8060754653051888,
    bridge:0.4846195714506714,
    ground-track-field:0.6266419749884015,
    small-vehicle:0.7576968365825171,
    large-vehicle:0.7464828428866678,
    ship:0.8704564379891551,
    tennis-court:0.8979494543163655,
    basketball-court:0.843057511295828,
    storage-tank:0.8425849319451546,
    soccer-ball-field:0.5014827828124655,
    roundabout:0.6093205806005668,
    harbor:0.6539914137566362,
    swimming-pool:0.6907127842035483,
    helicopter:0.6077007668477507

The submitted information is :

Description: FPN_Res50D_DOTA_KL_1x_20210704_35.1w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
