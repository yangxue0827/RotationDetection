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
REG_WEIGHT = 2.0
REG_LOSS_MODE = 1

VERSION = 'RetinaNet_DOTA_R3Det_GWD_NORM_2x_20220630'

"""
r3det+gwd norm
FLOPs: 1032047460;    Trainable params: 37769656
"""

