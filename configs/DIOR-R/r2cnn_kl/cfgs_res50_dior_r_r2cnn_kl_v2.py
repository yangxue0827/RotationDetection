# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
from configs._base_.models.faster_rcnn_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 11725 * 2
DECAY_EPOCH = [16, 22, 40]
MAX_EPOCH = 24
WARM_EPOCH = 1 / 16.
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DIOR-R'
CLASS_NUM = 20
IMG_SHORT_SIDE_LEN = [800, 450, 500, 640, 700, 900, 1000, 1100, 1200]
IMG_MAX_LENGTH = 1200
# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = True

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
LEVEL = ['P2', 'P3', 'P4', 'P5', 'P6']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [4, 8, 16, 32, 64]
ANCHOR_SCALES = [1.0]
ANCHOR_RATIOS = [0.5, 1., 2.0]

# loss
FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0

KL_TAU = 1.0
KL_FUNC = 1   # 0: sqrt  1: log

VERSION = 'FPN_Res50D_DIOR_R_R2CNN_KL_4x_20211028'

"""
R2CNN + KL
FLOPs: 1024386459;    Trainable params: 41791120
"""
