# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = '0'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 10.0
REG_LOSS_MODE = 4  # ProbIoU

VERSION = 'RetinaNet_DOTA_PROBIOU_1x_20210619'

"""
ProbIoU
FLOPs: 484911745;    Trainable params: 33002916
This is your result for task 1:

mAP: 0.6699477920098148
ap of each class: plane:0.8839667840781353, baseball-diamond:0.6914718502944607, bridge:0.41256920258313484, ground-track-field:0.6475098459337166, small-vehicle:0.7431085118997922, large-vehicle:0.6649908934311772, ship:0.8010632589243115, tennis-court:0.8983652865397685, basketball-court:0.7704836125997283, storage-tank:0.7808487163560258, soccer-ball-field:0.487485991493025, roundabout:0.59191194379331, harbor:0.5724836001531287, swimming-pool:0.6315901674107062, helicopter:0.4713672146567996
The submitted information is :

Description: RetinaNet_DOTA_PROBIOU_1x_20210619_35.1w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""