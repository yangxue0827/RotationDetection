# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

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
REG_LOSS_MODE = 3  # ProbIoU

VERSION = 'RetinaNet_DOTA_PROBIOU_1x_20210620'

"""
ProbIoU
FLOPs: 484911745;    Trainable params: 33002916
This is your result for task 1:

mAP: 0.6779227848901664
ap of each class: plane:0.882858174226469, baseball-diamond:0.756743334832494, bridge:0.419178004072785, ground-track-field:0.6666751258012132, small-vehicle:0.7149153735030855, large-vehicle:0.6060756694458895, ship:0.7541913337662483, tennis-court:0.8942261638430204, basketball-court:0.7759635370637713, storage-tank:0.7900775169387781, soccer-ball-field:0.551437401616416, roundabout:0.6243620785690698, harbor:0.6056204703575755, swimming-pool:0.6500206703470651, helicopter:0.47649691896861807
The submitted information is :

Description: RetinaNet_DOTA_PROBIOU_1x_20210620_35.1w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""