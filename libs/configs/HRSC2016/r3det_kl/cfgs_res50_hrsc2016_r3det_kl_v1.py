# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 10000 * 2
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

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

KL_TAU = 2.0
KL_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_HRSC2016_R3Det_KL_2x_20210205'

"""
r3det + kl + sqrt tau=2

07 metric
FLOPs: 1006485773;    Trainable params: 37059716
cls : ship|| Recall: 0.9714983713355049 || Precison: 0.4101065658301822|| AP: 0.8997363696966927
F1:0.9402011864106813 P:0.9523809523809523 R:0.9283387622149837
mAP is : 0.8997363696966927    444/444 [00:20<00:00, 22.08it/s]

89.97  89.86  89.73  89.51  87.34  77.38  61.17  25.12  3.76  0.13

12 metric
cls : ship|| Recall: 0.9714983713355049 || Precison: 0.4101065658301822|| AP: 0.9556716551331678
F1:0.9402011864106813 P:0.9523809523809523 R:0.9283387622149837
mAP is : 0.9556716551331678

"""

