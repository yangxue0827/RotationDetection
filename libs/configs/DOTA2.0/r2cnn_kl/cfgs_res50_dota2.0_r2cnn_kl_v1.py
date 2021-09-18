# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 40000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0

KL_TAU = 1.0
KL_FUNC = 1   # 0: sqrt  1: log

VERSION = 'FPN_Res50D_DOTA2.0_KL_1x_20210706'

"""
R2CNN + KLD
FLOPs: 1024596018;    Trainable params: 41791132

This is your evaluation result for task 1:

mAP: 0.5130343218305747
ap of each class:
plane:0.7931025663005032,
baseball-diamond:0.5208521894104177,
bridge:0.4260398742124911,
ground-track-field:0.6068816448736972,
small-vehicle:0.6464805846084523,
large-vehicle:0.5293451363632599,
ship:0.6712595590592587,
tennis-court:0.7707340150177826,
basketball-court:0.6059072889368493,
storage-tank:0.7363532565264833,
soccer-ball-field:0.39126720255687775,
roundabout:0.5596715552728614,
harbor:0.4521895092950472,
swimming-pool:0.6260269795369416,
helicopter:0.5238555951624132,
container-crane:0.01386748844375963,
airport:0.23016168047650185,
helipad:0.13062166689674687

The submitted information is :

Description: FPN_Res50D_DOTA2.0_KL_1x_20210706_52w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
