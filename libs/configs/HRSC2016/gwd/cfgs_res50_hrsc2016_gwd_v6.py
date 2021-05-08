# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = '3'
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
REG_WEIGHT = 1.0
REG_LOSS_MODE = 2  # GWD loss

GWD_TAU = 2.0
GWD_FUNC = tf.sqrt

VERSION = 'RetinaNet_HRSC2016_GWD_1x_20201220'

"""
RetinaNet-H + gwd fix bug (wasserstein_distance = tf.maximum(tf.log(wasserstein_distance + 1e-3), 0.0))
+ sqrt tau=2
FLOPs: 472715470;    Trainable params: 32325246
cls : ship|| Recall: 0.9397394136807817 || Precison: 0.6494091164884637|| AP: 0.8556218502153536
F1:0.8895116092874299 P:0.8748031496062992 R:0.9047231270358306
mAP is : 0.8556218502153536

85.56  84.97  84.04  75.99  73.32  60.31  44.39  17.14  3.13  0.06

SWA
cls : ship|| Recall: 0.9372964169381107 || Precison: 0.6355604638321369|| AP: 0.8533035469890232
F1:0.8935433875930093 P:0.8849840255591054 R:0.9022801302931596
mAP is : 0.8533035469890232

85.33  85.02  84.07  75.30  73.97  62.71  45.65  21.73  3.1  0.36
"""


