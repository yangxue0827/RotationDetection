# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_EPOCH = [36, 48, 60]
MAX_EPOCH = 51
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
IMG_SHORT_SIDE_LEN = [800, 450, 500, 640, 700, 900, 1000, 1100, 1200]
IMG_MAX_LENGTH = 1200
# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = True

# model
NET_NAME = 'resnet152_v1d'
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_SUBNET_CONV = 5

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA_BCD_6x_20210823'

"""
RetinaNet-H + 1-1/(sqrt(bcd)+2)
FLOPs: 1731833909;    Trainable params: 68720548

This is your result for task 1:

    mAP: 0.768573406585726
    ap of each class:
    plane:0.8880454724159974,
    baseball-diamond:0.8440907408590022,
    bridge:0.5373172162325457,
    ground-track-field:0.702561153438066,
    small-vehicle:0.778541230787683,
    large-vehicle:0.7630502779604436,
    ship:0.8518063491091422,
    tennis-court:0.9082940506398122,
    basketball-court:0.8590521594741499,
    storage-tank:0.856084607613421,
    soccer-ball-field:0.6477070835293367,
    roundabout:0.6414710867102008,
    harbor:0.765984221873397,
    swimming-pool:0.7719071490415356,
    helicopter:0.7126882991011564

The submitted information is :

Description: RetinaNet_DOTA_BCD_6x_20210823_275.4w_cpunms
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue


This is your result for task 1:

    mAP: 0.7851749645804825
    ap of each class:
    plane:0.8934950901089633,
    baseball-diamond:0.8634613422890471,
    bridge:0.547928015549363,
    ground-track-field:0.8097701712658131,
    small-vehicle:0.7778510118872703,
    large-vehicle:0.7482730257889372,
    ship:0.8418537972791365,
    tennis-court:0.9078547385450373,
    basketball-court:0.8626051059617449,
    storage-tank:0.8574141297978217,
    soccer-ball-field:0.7201907001872376,
    roundabout:0.6918555163756489,
    harbor:0.7720189968017138,
    swimming-pool:0.7639497267610552,
    helicopter:0.7191031001084477

The submitted information is :

Description: RetinaNet_DOTA_BCD_6x_20210823_275.4w_cpunms_mss
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui
"""
