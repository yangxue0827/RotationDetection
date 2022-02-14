# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 27000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_DOTA_KF_1x_20210902'

"""
RetinaNet-H + kfiou (exp(1-IoU)-1)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

This is your result for task 1:

    mAP: 0.6992391773413088
    ap of each class:
    plane:0.892002960041615,
    baseball-diamond:0.7732858199679905,
    bridge:0.461488701631356,
    ground-track-field:0.6568226387825112,
    small-vehicle:0.7191446514672439,
    large-vehicle:0.6496180915914522,
    ship:0.7756721605851568,
    tennis-court:0.8957877374986684,
    basketball-court:0.8218284672164252,
    storage-tank:0.7863219622742004,
    soccer-ball-field:0.5704904341914929,
    roundabout:0.6612983503314489,
    harbor:0.6307078540759418,
    swimming-pool:0.6814249757638241,
    helicopter:0.5126928547003046

The submitted information is :

Description: RetinaNet_DOTA_KF_1x_20210902_35.1w
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

This is your result for task 1:

    mAP: 0.7064133708777482
    ap of each class:
    plane:0.8949501034578764,
    baseball-diamond:0.7796459013723996,
    bridge:0.46303070580719785,
    ground-track-field:0.6726390727983068,
    small-vehicle:0.7256290420401867,
    large-vehicle:0.6561151622340775,
    ship:0.7815922299808545,
    tennis-court:0.9077429747421872,
    basketball-court:0.8301767177490947,
    storage-tank:0.7914564206644479,
    soccer-ball-field:0.5844254572326262,
    roundabout:0.6633575297546964,
    harbor:0.6368083141217261,
    swimming-pool:0.6855568710558761,
    helicopter:0.52307406015467

The submitted information is :

Description: RetinaNet_DOTA_KF_1x_20210902_35.1w_cpunms
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



