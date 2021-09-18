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
DATASET_NAME = 'ICDAR2015'
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1000
CLASS_NUM = 1

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
REG_LOSS_MODE = 3  # KLD loss

KL_TAU = 1.0
KL_FUNC = 0   # 0: sqrt  1: log

# post-processing
VIS_SCORE = 0.9

VERSION = 'RetinaNet_ICDAR2015_KL_1x_20210204'

"""
2021-02-05	kl	70.15%	81.31%	75.32%    0.9
2021-02-05  kl	71.55%	79.21%	75.18%    0.85
2021-02-05  kl	73.04%	77.12%	75.02%    0.8
2021-02-05  kl	73.95%	75.44%	74.69%    0.75
2021-02-05	kl	66.78%	84.06%	74.43%    0.95

"""

