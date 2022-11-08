# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

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
DATASET_NAME = 'ICDAR2015'
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1000
CLASS_NUM = 1

# data augmentation
IMG_ROTATE = True
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CENTER_LOSS_MODE = 0
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

# post-processing
VIS_SCORE = 0.73

VERSION = 'RetinaNet_ICDAR2015_KF_2x_20210903'

"""
loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

FLOPs: 472717841;    Trainable params: 32325246

Hmean50:95: 0.7736086628053386  0.7423822714681441  0.707126668345505  0.6441702341979352  0.5469654998740872
            0.39536640644673887  0.22916142029715436  0.09317552253840343  0.01863510450768069  0.000503651473180559

0.73: Calculated!{"precision": 0.8109820485744457, "recall": 0.7395281656234954, "hmean": 0.7736086628053386, "AP": 0}%
0.72: Calculated!{"precision": 0.8061650992685475, "recall": 0.7428984111699567, "hmean": 0.7732397895264345, "AP": 0}%
0.74: Calculated!{"precision": 0.8162828066416711, "recall": 0.7337506018295619, "hmean": 0.7728194726166329, "AP": 0}%
0.75: Calculated!{"precision": 0.8206933911159263, "recall": 0.7294174289841117, "hmean": 0.7723680856487383, "AP": 0}%
0.8: Calculated!{"precision": 0.842741935483871, "recall": 0.7043813192103996, "hmean": 0.7673747705218987, "AP": 0}%
0.83: Calculated!{"precision": 0.8530293941211757, "recall": 0.6846413095811267, "hmean": 0.7596153846153846, "AP": 0}%
0.85: Calculated!{"precision": 0.8652526512788522, "recall": 0.6677900818488204, "hmean": 0.753804347826087, "AP": 0}%
0.86: Calculated!{"precision": 0.8701464035646085, "recall": 0.6581608088589311, "hmean": 0.7494517543859649, "AP": 0}%
"""

