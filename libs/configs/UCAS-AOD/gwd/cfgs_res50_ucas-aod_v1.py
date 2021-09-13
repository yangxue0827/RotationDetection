# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = '0'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 5000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'UCAS-AOD'
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1500
CLASS_NUM = 2

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
REG_LOSS_MODE = 2  # GWD loss

GWD_TAU = 2.0
GWD_FUNC = tf.sqrt

# test and eval
USE_07_METRIC = False

VERSION = 'RetinaNet_UCAS-AOD_2x_20201223'

"""
USE_07_METRIC
cls : plane|| Recall: 0.9774266365688488 || Precison: 0.6141843971631206|| AP: 0.9029647561346286
F1:0.9722979109900091 P:0.9780721790772042 R:0.9665914221218962
cls : car|| Recall: 0.9662775616083009 || Precison: 0.46101485148514854|| AP: 0.8889943787746151
F1:0.9182954181352132 P:0.9075364154528183 R:0.9293125810635539
mAP is : 0.8959795674546218

USE_12_METRIC
cls : car|| Recall: 0.9662775616083009 || Precison: 0.46087225487163624|| AP: 0.9402692398957737
F1:0.9183477425552354 P:0.9070208728652751 R:0.9299610894941635
cls : plane|| Recall: 0.9774266365688488 || Precison: 0.6141843971631206|| AP: 0.9685903833809424
F1:0.9722979109900091 P:0.9780721790772042 R:0.9665914221218962
mAP is : 0.954429811638358
"""

