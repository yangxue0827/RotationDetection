# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 5.0

VERSION = 'RetinaNet_DOTA_R3Det_KF_2x_20210913'

"""
r3det + kfiou -ln(IoU)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

FLOPs: 1032041742;    Trainable params: 37769656

This is your result for task 1:

mAP: 0.7227739479718984
ap of each class: plane:0.8929036567485278, baseball-diamond:0.7961617602531166, bridge:0.4653266947623756, ground-track-field:0.6684974406801223, small-vehicle:0.7677301682588333, large-vehicle:0.787021541820863, ship:0.8705983375078381, tennis-court:0.9069806516071823, basketball-court:0.8125219847780131, storage-tank:0.8076742566630796, soccer-ball-field:0.5869593577552126, roundabout:0.613896899239834, harbor:0.6086345093441838, swimming-pool:0.6844993929872109, helicopter:0.5722025671720847
The submitted information is :

Description: RetinaNet_DOTA_R3Det_KF_2x_20210913_70.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

"""

