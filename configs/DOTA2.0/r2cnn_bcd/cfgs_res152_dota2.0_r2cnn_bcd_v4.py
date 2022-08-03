# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
from configs._base_.models.faster_rcnn_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 32000
DECAY_EPOCH = [24, 32, 40]
MAX_EPOCH = 34
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18
IMG_SHORT_SIDE_LEN = [800, 450, 500, 640, 700, 900, 1000, 1100, 1200]
IMG_MAX_LENGTH = 1200
# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = True

# model
# backbone
NET_NAME = 'resnet152_v1d'
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

# rpn sample
RPN_TOP_K_NMS_TEST = 12000
RPN_MAXIMUM_PROPOSAL_TEST = 3000

VERSION = 'FPN_Res152D_DOTA2.0_BCD_2x_20211022'

"""
R2CNN + BCD
FLOPs: 1467045885;    Trainable params: 76328604

This is your evaluation result for task 1:

    mAP: 0.5867621893205778
    ap of each class: plane:0.8418050358374268,
    baseball-diamond:0.573905808296434,
    bridge:0.47966653680549726,
    ground-track-field:0.6856825773132934,
    small-vehicle:0.5749652321917142,
    large-vehicle:0.5648841624550318,
    ship:0.6792455805387367,
    tennis-court:0.8188892234690702,
    basketball-court:0.6751462691220294,
    storage-tank:0.7269858934601133,
    soccer-ball-field:0.5388814842389147,
    roundabout:0.6201073048067846,
    harbor:0.52203628363618,
    swimming-pool:0.6386854577787894,
    helicopter:0.7221891835217363,
    container-crane:0.17765077259126374,
    airport:0.5178594057710232,
    helipad:0.20313319593636103

The submitted information is :

Description: FPN_Res152D_DOTA2.0_BCD_2x_20211022_108.8w_mss
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
