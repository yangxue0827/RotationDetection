# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.0003
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

# neck
FPN_CHANNEL = 256

# rpn head
ANCHOR_MODE = 'R'
ANCHOR_RATIOS = [0.5, 1., 2.0, 1/4.0, 4.0]

# roi sample
FAST_RCNN_MINIBATCH_SIZE = 256

# loss
RPN_LOCATION_LOSS_WEIGHT = 1 / 7
RPN_CLASSIFICATION_LOSS_WEIGHT = 2.0
FAST_RCNN_LOCATION_LOSS_WEIGHT = 4.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 2.0

VERSION = 'FPN_Res50D_DOTA_RRPN_1x_20210601'

"""
RRPN
FLOPs: 1209944239;    Trainable params: 41215026
"""
