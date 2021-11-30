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
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
CTR_WEIGHT = 1.0
REG_LOSS_MODE = 0

VERSION = 'FCOS_DOTA_RSDet_1x_20210622'

"""
FCOS + modulated loss
FLOPs: 468484120;    Trainable params: 32090136

This is your result for task 1:

    mAP: 0.6790645938261939
    ap of each class:
    plane:0.8729861339230366,
    baseball-diamond:0.7538884620192762,
    bridge:0.4587526572337119,
    ground-track-field:0.5802029845851726,
    small-vehicle:0.7337562383425493,
    large-vehicle:0.634861376842494,
    ship:0.7677495389605099,
    tennis-court:0.8914462748625346,
    basketball-court:0.7506157206033095,
    storage-tank:0.8285886854886657,
    soccer-ball-field:0.5257314976581912,
    roundabout:0.6079512703389356,
    harbor:0.6050630359585935,
    swimming-pool:0.6476218493173386,
    helicopter:0.5267531812585877

The submitted information is :

Description: FCOS_DOTA_RSDet_1x_20210622_35.1w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



