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
SAVE_WEIGHTS_INTE = 5000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'MSRA-TD500'
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
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

# post-processing
VIS_SCORE = 0.85

VERSION = 'RetinaNet_MSRA_TD500_KF_2x_20210903'

"""
loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

FLOPs: 472717841;    Trainable params: 32325246

Hmean50:95: 0.763016157989228  0.7360861759425494  0.6983842010771993  0.658886894075404  0.5709156193895871
            0.4757630161579892  0.3393177737881508  0.19210053859964096  0.05745062836624775  0.0035906642728904844

0.99: Calculated!{"precision": 0.7988721804511278, "recall": 0.7302405498281787, "hmean": 0.763016157989228, "AP": 0}%
0.992: Calculated!{"precision": 0.8065134099616859, "recall": 0.7233676975945017, "hmean": 0.7626811594202898, "AP": 0}%
0.98: Calculated!{"precision": 0.783001808318264, "recall": 0.7439862542955327, "hmean": 0.7629955947136565, "AP": 0}%
0.97: Calculated!{"precision": 0.7674825174825175, "recall": 0.7542955326460481, "hmean": 0.7608318890814558, "AP": 0}%
0.96: Calculated!{"precision": 0.7555555555555555, "recall": 0.7594501718213058, "hmean": 0.7574978577549271, "AP": 0}%
0.95: Calculated!{"precision": 0.7504244482173175, "recall": 0.7594501718213058, "hmean": 0.7549103330486763, "AP": 0}%
0.9: Calculated!{"precision": 0.7311475409836066, "recall": 0.7663230240549829, "hmean": 0.7483221476510067, "AP": 0}%
0.85: Calculated!{"precision": 0.710236220472441, "recall": 0.7749140893470791, "hmean": 0.7411668036154477, "AP": 0}%

"""

