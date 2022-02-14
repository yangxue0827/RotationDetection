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

VERSION = 'RetinaNet_DOTA1.5_R3Det_KF_2x_20210912'

"""
r3det + kfiou (exp(1-IoU)-1)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

FLOPs: 1033872993;    Trainable params: 37820366

This is your evaluation result for task 1:

mAP: 0.6469309615389345
ap of each class: plane:0.8019585893797351, baseball-diamond:0.7755837639009221, bridge:0.4703781861580584, ground-track-field:0.6693702607603237, small-vehicle:0.5684469094725499, large-vehicle:0.7488156179457427, ship:0.8434063774366767, tennis-court:0.9086226435635673, basketball-court:0.767724830376888, storage-tank:0.6708704453223998, soccer-ball-field:0.4701937280870145, roundabout:0.7039770827458453, harbor:0.573665309268502, swimming-pool:0.6636100952513613, helicopter:0.5722260904079086, container-crane:0.14204545454545453
The submitted information is :

Description: RetinaNet_DOTA1.5_R3Det_KF_2x_20210912_83.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

