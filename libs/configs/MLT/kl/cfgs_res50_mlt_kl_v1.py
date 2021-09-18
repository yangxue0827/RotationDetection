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
DATASET_NAME = 'MLT'
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1000
CLASS_NUM = 1

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
REG_LOSS_MODE = 3  # KLD loss

KL_TAU = 2.0
KL_FUNC = 0   # 0: sqrt  1: log

# post-processing
VIS_SCORE = 0.2

VERSION = 'RetinaNet_MLT_KL_1x_20210206'

"""
2021-02-07	kl_0.35	57.59%	72.50%	47.77%	43.95%
2021-02-07	kl_0.4	57.49%	75.41%	46.45%	42.97%
2021-02-07	kl_0.3	57.23%	68.86%	48.96%	44.79%
2021-02-07  kl_0.45	57.19%	78.06%	45.12%	41.96%

"""

