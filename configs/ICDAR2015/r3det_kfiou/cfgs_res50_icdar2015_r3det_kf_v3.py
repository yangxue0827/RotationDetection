# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0'
NUM_GPU = len(GPU_GROUP.strip().split(','))
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

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# post-processing
VIS_SCORE = 0.7

VERSION = 'RetinaNet_ICDAR2015_R3Det_KF_1x_20210807'

"""
FLOPs: 557923743;    Trainable params: 37059716
0.7: Calculated!{"precision": 0.7911857292759706, "recall": 0.7260471834376505, "hmean": 0.7572181772533267, "AP": 0}%
0.75: Calculated!{"precision": 0.8066630256690334, "recall": 0.7111218103033221, "hmean": 0.7558853633572159, "AP": 0}%

0.7572181772533267  0.725081596786342  0.6763745920160683  0.6030630178257594  0.48958071805171977
0.345970374089882  0.18980667838312829  0.0718051719809189  0.010042681395932715  0.0005021340697966357

"""

