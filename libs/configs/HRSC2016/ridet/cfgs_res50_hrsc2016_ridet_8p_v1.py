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
DATASET_NAME = 'HRSC2016'
IMG_SHORT_SIDE_LEN = 512
IMG_MAX_LENGTH = 512
CLASS_NUM = 1

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
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_HRSC2016_RIDet_1x_20210507'

"""
RIDet-8p
FLOPs: 475328945;    Trainable params: 32470461
cls : ship|| Recall: 0.9820846905537459 || Precison: 0.3577573420350044|| AP: 0.8274989342996861
F1:0.8383138431933053 P:0.7834394904458599 R:0.9014657980456026
mAP is : 0.8274989342996861

AP50:95: 82.79  82.14  80.63  79.88  71.72  68.04  56.41  35.76  12.85 0.34
"""


