# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 40000
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
REG_WEIGHT = 1.0
CTR_WEIGHT = 1.0
REG_LOSS_MODE = None

VERSION = 'FCOS_DOTA2.0_1x_20210617'

"""
FCOS
FLOPs: 468608567;    Trainable params: 32097051

This is your evaluation result for task 1:

    mAP: 0.4809972885325565
    ap of each class:
    plane:0.7659284378972354,
    baseball-diamond:0.46139160876113167,
    bridge:0.3916619664774063,
    ground-track-field:0.5363254969797883,
    small-vehicle:0.47953744611698595,
    large-vehicle:0.4525891200331043,
    ship:0.5553860292941958,
    tennis-court:0.7715844969314888,
    basketball-court:0.5532847655452562,
    storage-tank:0.6472592062532458,
    soccer-ball-field:0.41866827138746154,
    roundabout:0.5188837390650718,
    harbor:0.4341723234725953,
    swimming-pool:0.550704924084627,
    helicopter:0.39736327929925497,
    container-crane:0.12736304462923168,
    airport:0.4779544132916722,
    helipad:0.11789262406626426

The submitted information is :

Description: FCOS_DOTA2.0_1x_20210617_52w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



