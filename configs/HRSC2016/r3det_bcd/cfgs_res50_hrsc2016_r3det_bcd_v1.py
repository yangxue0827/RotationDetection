# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 10000 * 2
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

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_HRSC2016_R3Det_BCD_2x_20210812'

"""
r3det + bcd + sqrt tau=2

07 metric
cls : ship|| Recall: 0.9731270358306189 || Precison: 0.2994237033324981|| AP: 0.9006499760418762
F1:0.9393392625366898 P:0.9455445544554455 R:0.9332247557003257
mAP is : 0.9006499760418762

12 metric
cls : ship|| Recall: 0.9731270358306189 || Precison: 0.2994237033324981|| AP: 0.9564135316767767
F1:0.9393392625366898 P:0.9455445544554455 R:0.9332247557003257
mAP is : 0.9564135316767767

0.9006499760418762  0.8999341168041491  0.897464476725123  0.8920394732699007  0.8642395416893606
0.7624070922084628  0.5372873768407043  0.23418739181864964  0.03773701708127938  0.0004614674665436086

"""

