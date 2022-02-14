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

VERSION = 'RetinaNet_DOTA_R3Det_KF_2x_20210915'

"""
r3det + kfiou (1-IoU)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

FLOPs: 1032047562;    Trainable params: 37769656

This is your result for task 1:

mAP: 0.7109249523397371
ap of each class: plane:0.8931613743199684, baseball-diamond:0.766112730971134, bridge:0.46608740346050387, ground-track-field:0.6536008894337909, small-vehicle:0.7689938681045967, large-vehicle:0.7680716510302534, ship:0.8697074674460797, tennis-court:0.9078757137302751, basketball-court:0.7938536725079461, storage-tank:0.8201423975604443, soccer-ball-field:0.5376793509723141, roundabout:0.6105185692799017, harbor:0.5763521972539241, swimming-pool:0.6953691446479559, helicopter:0.5363478543769697
The submitted information is :

Description: RetinaNet_DOTA_R3Det_KF_2x_20210915_70.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

