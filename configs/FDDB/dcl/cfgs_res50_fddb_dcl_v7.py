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
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 0.5

# DCL
OMEGA = 180 / 64.
ANGLE_MODE = 1  # {0: BCL, 1: GCL}

# eval
USE_07_METRIC = False

VERSION = 'RetinaNet_FDDB_DCL_G_2x_20211107'

"""
FLOPs: 833819040;    Trainable params: 32263011

2007
cls : face|| Recall: 0.9731404958677686 || Precison: 0.6399456521739131|| AP: 0.9075850736890454
F1:0.9549562972685337 P:0.9762589928057553 R:0.9345730027548209
mAP is : 0.9075850736890454

2012
cls : face|| Recall: 0.9731404958677686 || Precison: 0.6399456521739131|| AP: 0.9676994385070845
F1:0.9549562972685337 P:0.9762589928057553 R:0.9345730027548209
mAP is : 0.9676994385070845

AP50:95
0.9676994385070845  0.9570325130462717  0.9229211707137834  0.868744159294526  0.7927268276839632
0.6835083860211622  0.49790244178542364  0.2570679655371121  0.043521157950894396  0.00022918761076853254

"""

