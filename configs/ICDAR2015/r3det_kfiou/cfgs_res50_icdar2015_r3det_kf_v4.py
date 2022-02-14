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

# post-processing
VIS_SCORE = 0.65

VERSION = 'RetinaNet_ICDAR2015_R3Det_KF_2x_20210807'

"""
0.65: Calculated!{"precision": 0.8374116367591082, "recall": 0.7414540202214733, "hmean": 0.7865168539325843, "AP": 0}%
0.64: Calculated!{"precision": 0.8318965517241379, "recall": 0.7433798748194511, "hmean": 0.7851512840071192, "AP": 0}%
0.67: Calculated!{"precision": 0.8439241917502787, "recall": 0.7289359653346172, "hmean": 0.7822268147765434, "AP": 0}%
0.63: Calculated!{"precision": 0.8275493860117459, "recall": 0.746268656716418, "hmean": 0.7848101265822784, "AP": 0}%

0.7865168539325843  0.7599591419816139  0.7114402451481104  0.6542390194075587  0.5469867211440246
0.39427987742594484  0.21756894790602657  0.0888661899897855  0.015321756894790602  0.0010214504596527067

"""

