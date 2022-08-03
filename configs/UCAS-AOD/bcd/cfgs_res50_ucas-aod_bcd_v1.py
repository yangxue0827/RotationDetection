# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

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
BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

# test and eval
USE_07_METRIC = False

VERSION = 'RetinaNet_UCAS-AOD_BCD_2x_20210818'

"""

USE_07_METRIC
cls : plane|| Recall: 0.9909706546275395 || Precison: 0.5227435103596094|| AP: 0.9028055670119146
F1:0.9764442754381748 P:0.97955474784189 R:0.9733634311512416
cls : car|| Recall: 0.9831387808041504 || Precison: 0.24979403526116328|| AP: 0.8871291990659329
F1:0.9230719232399923 P:0.927916120576671 R:0.9182879377431906
mAP is : 0.8949673830389238

USE_12_METRIC
9F1:0.9230719232399923 P:0.927916120576671 R:0.9182879377431906
cls : plane|| Recall: 0.9909706546275395 || Precison: 0.5227435103596094|| AP: 0.9809507102258819
F1:0.9764442754381748 P:0.97955474784189 R:0.9733634311512416
mAP is : 0.965426254607672
"""
