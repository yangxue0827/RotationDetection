# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 32000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16

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

VERSION = 'RetinaNet_DOTA1.5_R3Det_KF_2x_20210915'

"""
r3det + kfiou -ln(IoU)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

FLOPs: 1033867173;    Trainable params: 37820366

This is your evaluation result for task 1:

mAP: 0.645403626015407
ap of each class: plane:0.8053047262958998, baseball-diamond:0.753655771381029, bridge:0.46478831301089446, ground-track-field:0.666421869032318, small-vehicle:0.5684836225650711, large-vehicle:0.7486230321017128, ship:0.846294510406009, tennis-court:0.907114490451818, basketball-court:0.7419085772260162, storage-tank:0.6709156258251443, soccer-ball-field:0.5108195563344049, roundabout:0.6736387259641492, harbor:0.6242607432347067, swimming-pool:0.6618725660955358, helicopter:0.5343332523210572, container-crane:0.14802263400074617
The submitted information is :

Description: RetinaNet_DOTA1.5_R3Det_KF_2x_20210915_83.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

