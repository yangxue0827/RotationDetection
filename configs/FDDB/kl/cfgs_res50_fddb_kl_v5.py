# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 2000 * 2
DECAY_EPOCH = [8, 11, 20]
MAX_EPOCH = 12
WARM_EPOCH = 1 / 16.
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'FDDB'
CLASS_NUM = 1

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_SUBNET_CONV = 4
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [1, 1 / 1.5, 1.5]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0
REG_LOSS_MODE = 3  # KLD loss

KL_TAU = 1.0
KL_FUNC = 1   # 0: sqrt  1: log

VERSION = 'RetinaNet_FDDB_KL_2x_20211107'

"""
RetinaNet-H + kl (fix bug)

2007
cls : face|| Recall: 0.9800275482093664 || Precison: 0.5909468438538206|| AP: 0.9086022298878069
F1:0.9599106140657935 P:0.9806034482758621 R:0.9400826446280992
mAP is : 0.9086022298878069

2012
cls : face|| Recall: 0.9800275482093664 || Precison: 0.5909468438538206|| AP: 0.975114766801636
F1:0.9599106140657935 P:0.9806034482758621 R:0.9400826446280992
mAP is : 0.975114766801636

AP50:95
0.975114766801636  0.9666442174651226  0.9540159408453797  0.9323120706611414  0.9064024625303708
0.8532868702313012  0.6871896787782661  0.4220118463759825  0.10333811036980581  0.000880994697104151
"""
