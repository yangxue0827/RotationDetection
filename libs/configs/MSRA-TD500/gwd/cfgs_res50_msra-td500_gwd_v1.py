# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = '0'
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
REG_LOSS_MODE = 2  # GWD loss

GWD_TAU = 2.0
GWD_FUNC = tf.sqrt

# post-processing
VIS_SCORE = 0.8

VERSION = 'RetinaNet_MSRA_TD500_GWD_2x_20210223'

"""

0.80: Calculated!{"precision": 0.7777777777777778, "recall": 0.7577319587628866, "hmean": 0.7676240208877285, "AP": 0}%
0.85: Calculated!{"precision": 0.786618444846293, "recall": 0.7474226804123711, "hmean": 0.7665198237885463, "AP": 0}%
0.75: Calculated!{"precision": 0.7700348432055749, "recall": 0.7594501718213058, "hmean": 0.7647058823529411, "AP": 0}%

Hmean50:95 =  0.767624020887728 0.7293298520452567 0.6858137510879024 0.6248912097476066 0.5622280243690165
              0.44212358572671884 0.3115752828546562 0.1775456919060052 0.04873803307223674 0.01218450826805918
0.4362053959965186
"""


