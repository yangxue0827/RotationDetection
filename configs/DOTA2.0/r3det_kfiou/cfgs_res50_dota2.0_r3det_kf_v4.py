# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2,3'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 40000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

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

VERSION = 'RetinaNet_DOTA2.0_R3Det_KF_2x_20210912'

"""
r3det + kfiou (exp(1-IoU)-1)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

FLOPs: 1269567351;    Trainable params: 37921786

This is your evaluation result for task 1:

mAP: 0.5041408791213118
ap of each class: plane:0.7956770809019875, baseball-diamond:0.4944565683803126, bridge:0.40319549402405697, ground-track-field:0.5866262451680978, small-vehicle:0.43356455097397906, large-vehicle:0.5267194446774651, ship:0.5981167704578747, tennis-court:0.779570922426722, basketball-court:0.6127302248493062, storage-tank:0.5785358709898982, soccer-ball-field:0.44872732145129846, roundabout:0.5019848645775529, harbor:0.4363876875131218, swimming-pool:0.5622704109974263, helicopter:0.5521002590451737, container-crane:0.1489275613805301, airport:0.4650029265863282, helipad:0.1499416197824809
The submitted information is :

Description: RetinaNet_DOTA2.0_R3Det_KF_2x_20210912_104w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

