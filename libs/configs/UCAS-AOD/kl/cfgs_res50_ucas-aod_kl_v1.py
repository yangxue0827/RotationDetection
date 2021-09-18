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
SAVE_WEIGHTS_INTE = 5000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'UCAS-AOD'
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1500
CLASS_NUM = 2

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
REG_LOSS_MODE = 3  # KLD loss

KL_TAU = 2.0
KL_FUNC = 1   # 0: sqrt  1: log

# test and eval
USE_07_METRIC = False

VERSION = 'RetinaNet_UCAS-AOD_KL_2x_20210207'

"""
FLOPs: 473586635;    Trainable params: 32373651

USE_07_METRIC
cls : plane|| Recall: 0.9869074492099322 || Precison: 0.6047026279391424|| AP: 0.9042635914888404
cls : car|| Recall: 0.9760051880674449 || Precison: 0.4168975069252078|| AP: 0.8852959388679364
mAP is : 0.8947797651783884

USE_12_METRIC
cls : car|| Recall: 0.9760051880674449 || Precison: 0.4168975069252078|| AP: 0.9433789394477989
cls : plane|| Recall: 0.9869074492099322 || Precison: 0.6047026279391424|| AP: 0.9794079633322677
mAP is : 0.9613934513900333
"""
