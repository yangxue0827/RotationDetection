# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
from configs._base_.models.faster_rcnn_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

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

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'FPN_Res50D_DOTA_BCD_1x_20210818'

"""
R2CNN + BCD
This is your result for task 1:

mAP: 0.7160784929993794
ap of each class: plane:0.8901291908737273, baseball-diamond:0.7798384710011358, bridge:0.46930016097891625, ground-track-field:0.6681497793141755, small-vehicle:0.7089535115682858, large-vehicle:0.7451816419196996, ship:0.8583820607200123, tennis-court:0.8935909397555922, basketball-court:0.8243845550280713, storage-tank:0.8389692233579369, soccer-ball-field:0.49262597649134277, roundabout:0.6244218848846791, harbor:0.6361810262965861, swimming-pool:0.7031571782436455, helicopter:0.6079117945568833
The submitted information is :

Description: FPN_Res50D_DOTA_BCD_1x_20210818_35.1w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

"""
