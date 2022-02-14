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

VERSION = 'RetinaNet_ICDAR2015_R3Det_KF_2x_20210731'

"""
0.65: Calculated!{"precision": 0.7956278596847992, "recall": 0.7534906114588349, "hmean": 0.7739861523244314, "AP": 0}%
0.7: Calculated!{"precision": 0.8170144462279294, "recall": 0.7351949927780452, "hmean": 0.7739483020780537, "AP": 0}%
0.63: Calculated!{"precision": 0.7868525896414342, "recall": 0.7607125662012518, "hmean": 0.7735618115055081, "AP": 0}%
0.67: Calculated!{"precision": 0.8035251425609123, "recall": 0.746268656716418, "hmean": 0.7738392411382925, "AP": 0}%
0.73: Calculated!{"precision": 0.8270181219110379, "recall": 0.7250842561386616, "hmean": 0.7727039507439714, "AP": 0}%

Hmean50:95: 0.7739861523244314  0.7462908011869436  0.7047477744807121  0.6330365974282888  0.5158259149357071
            0.3644906033630069  0.20474777448071219  0.08110781404549952  0.013353115727002969  0

"""

