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
REG_WEIGHT = 0.01

# eval
USE_07_METRIC = False

VERSION = 'RetinaNet_FDDB_KF_2x_20211025'

"""
RetinaNet-H + KF
FLOPs: 830085199;    Trainable params: 32159286

2007
cls : face|| Recall: 0.9772727272727273 || Precison: 0.7295629820051414|| AP: 0.9080203490948332
F1:0.9570674865717835 P:0.9784172661870504 R:0.9366391184573003
mAP is : 0.9080203490948332

2012
cls : face|| Recall: 0.9772727272727273 || Precison: 0.7295629820051414|| AP: 0.9725235574788025
F1:0.9570674865717835 P:0.9784172661870504 R:0.9366391184573003
mAP is : 0.9725235574788025

AP50:95=0.6324882510834976
0.9725235574788025  0.9646017106880204  0.9489356162098821  0.9238151722266383  0.8785580846457443
0.7738287067786952  0.5676342067649506  0.25623565459409353  0.038508473303913306  0.00024132814423412932
"""
