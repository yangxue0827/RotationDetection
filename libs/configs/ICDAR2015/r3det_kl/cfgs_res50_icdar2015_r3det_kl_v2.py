# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0'
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 10000 * 2
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

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
USE_IOU_FACTOR = False

KL_TAU = 1.0
KL_FUNC = 0   # 0: sqrt  1: log

# post-processing
VIS_SCORE = 0.67

VERSION = 'RetinaNet_ICDAR2015_R3Det_KL_2x_20210207'

"""
FLOPs: 557923752;    Trainable params: 37059716
2021-02-08	r3det_kl	75.73%	83.94%	79.63%  0.67
2021-02-08	r3det_kl	75.97%	83.62%	79.62%  0.66
2021-02-08  r3det_kl	76.41%	83.09%	79.61%  0.65
2021-02-08	r3det_kl	75.35%	84.37%	79.60%  0.68
2021-02-08	r3det_kl	74.43%	85.32%	79.51%  0.7
2021-02-08	r3det_kl	76.60%	82.39%	79.39%  0.64
2021-02-08	r3det_kl	78.09%	79.74%	78.91%  0.6
"""

