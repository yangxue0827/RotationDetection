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
SAVE_WEIGHTS_INTE = 32000
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

# loss
CENTER_LOSS_MODE = 0
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_DOTA1.5_KF_1x_20210904'

"""
RetinaNet-H + kfiou (exp(1-IoU)-1)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

FLOPs: 485785312;    Trainable params: 33051321

This is your evaluation result for task 1:

    mAP: 0.621280142030018
    ap of each class:
    plane:0.7961673552273434,
    baseball-diamond:0.7919297962108256,
    bridge:0.4266645800680319,
    ground-track-field:0.6725785853434287,
    small-vehicle:0.49442312033891705,
    large-vehicle:0.617605960472413,
    ship:0.7589145831913362,
    tennis-court:0.8948022695795292,
    basketball-court:0.746940541069233,
    storage-tank:0.618297423618113,
    soccer-ball-field:0.5107664025542281,
    roundabout:0.6868455436184394,
    harbor:0.6161102848881593,
    swimming-pool:0.6437627881067691,
    helicopter:0.5731867467072281,
    container-crane:0.09148629148629149

The submitted information is :

Description: RetinaNet_DOTA1.5_KF_1x_20210904_41.6w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

This is your evaluation result for task 1:

    mAP: 0.62712466195339
    ap of each class:
    plane:0.7996320679842436,
    baseball-diamond:0.7931046992887474,
    bridge:0.4311646095372172,
    ground-track-field:0.681974776888092,
    small-vehicle:0.500308973139615,
    large-vehicle:0.6235146979358908,
    ship:0.7652515264086934,
    tennis-court:0.9081722581874745,
    basketball-court:0.7604111462221048,
    storage-tank:0.622222185541879,
    soccer-ball-field:0.5132890348503273,
    roundabout:0.6924873883332214,
    harbor:0.6229200343597264,
    swimming-pool:0.6513720079418596,
    helicopter:0.5766821592638252,
    container-crane:0.09148702537132289

The submitted information is :

Description: RetinaNet_DOTA1.5_KF_1x_20210904_41.6w_cpunms
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



