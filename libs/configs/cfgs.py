# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 5000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'ICDAR2015'
IMG_SHORT_SIDE_LEN = [1024, 512, 640, 768, 896, 1152, 1280]
IMG_MAX_LENGTH = 1536
CLASS_NUM = 1

# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = True

# model
# backbone
NET_NAME = 'resnet101_v1d'
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
ANGLE_WEIGHT = 1.0

# CSL
LABEL_TYPE = 0  # {0: gaussian_label, 1: rectangular_label, 2: pulse_label, 3: triangle_label}
RADUIUS = 6
OMEGA = 1

VERSION = 'FPN_Res101D_ICDAR2015_R2CNN_CSL_2x_20210324'

"""
R2CNN + CSL
FLOPs: 1170458120;    Trainable params: 60718736
"""

