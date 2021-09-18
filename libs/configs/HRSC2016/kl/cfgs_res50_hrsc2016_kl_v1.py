# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 10000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'HRSC2016'
IMG_SHORT_SIDE_LEN = 512
IMG_MAX_LENGTH = 512
CLASS_NUM = 1

# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
REG_LOSS_MODE = 3  # KLD loss

KL_TAU = 2.0
KL_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_HRSC2016_KL_1x_20210204'

"""
RetinaNet-H + kl + sqrt tau=2
cls : ship|| Recall: 0.9617263843648208 || Precison: 0.6605145413870246|| AP: 0.8745101996155792
F1:0.9070372542296279 P:0.8965791567223548 R:0.9177524429967426
mAP is : 0.8745101996155792   444/444 [00:21<00:00, 21.04it/s]

87.45  87.03  86.72  85.45  76.60  72.39  50.65  27.68  3.91  0.17

"""


