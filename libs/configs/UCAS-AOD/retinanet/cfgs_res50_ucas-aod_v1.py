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
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
REG_LOSS_MODE = None

# test and eval
USE_07_METRIC = False

VERSION = 'RetinaNet_UCAS-AOD_2x_20201224'

"""
FLOPs: 473586618;    Trainable params: 32373651
USE_07_METRIC
cls : car|| Recall: 0.9643320363164721 || Precison: 0.34183908045977013|| AP: 0.8787961248457516
cls : plane|| Recall: 0.9765237020316027 || Precison: 0.6103273137697517|| AP: 0.899666489478612
mAP is : 0.8892313071621818

USE_12_METRIC
cls : car|| Recall: 0.9643320363164721 || Precison: 0.34183908045977013|| AP: 0.9262148128348662
cls : plane|| Recall: 0.9765237020316027 || Precison: 0.6103273137697517|| AP: 0.9649656698273091
mAP is : 0.9455902413310877
"""
