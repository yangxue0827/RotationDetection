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

VERSION = 'RetinaNet_HRSC2016_KF_1x_20210726_v2'


"""
RetinaNet-H + kfiou (exp(1-IoU)-1)

loss_1 [n,]
loss_2 [n,]
loss = sum(loss_1+loss_2)/n

FLOPs: 472715480;    Trainable params: 32325246

0.5
cls : ship|| Recall: 0.9755700325732899 || Precison: 0.40734444066644|| AP: 0.838630675200037
F1:0.8651329816071854 P:0.8275092936802974 R:0.9063517915309446
mAP is : 0.838630675200037

0.75
cls : ship|| Recall: 0.75814332247557 || Precison: 0.31655899353961237|| AP: 0.5906449417795012
F1:0.7039821129907923 P:0.6964143426294821 R:0.7117263843648208
mAP is : 0.5906449417795012

0.838630675200037  0.8288021363476303  0.8177286823724391  0.7847955106832243  0.7075944990671382
0.5906449417795012  0.38525964134287144  0.19399886140633363  0.03692054278766962  0.0004741021690174232
"""


