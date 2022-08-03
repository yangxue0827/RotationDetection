# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 5000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'MSRA-TD500'
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

# loss
BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

# post-processing
VIS_SCORE = 0.93

VERSION = 'RetinaNet_MSRA_TD500_BCD_2x_20210723'

"""
0.75: Calculated!{"precision": 0.703416149068323, "recall": 0.7783505154639175, "hmean": 0.7389885807504079, "AP": 0}%
0.80: Calculated!{"precision": 0.7151898734177216, "recall": 0.7766323024054983, "hmean": 0.744645799011532, "AP": 0}%
0.85: Calculated!{"precision": 0.7300813008130081, "recall": 0.7714776632302406, "hmean": 0.7502088554720134, "AP": 0}%
0.90: Calculated!{"precision": 0.752542372881356, "recall": 0.7628865979381443, "hmean": 0.757679180887372, "AP": 0}%
0.92: Calculated!{"precision": 0.7577319587628866, "recall": 0.7577319587628866, "hmean": 0.7577319587628866, "AP": 0}
0.93: Calculated!{"precision": 0.7603448275862069, "recall": 0.7577319587628866, "hmean": 0.7590361445783133, "AP": 0}%
0.95: Calculated!{"precision": 0.7609841827768014, "recall": 0.7439862542955327, "hmean": 0.7523892267593397, "AP": 0}%

Hmean50:95 = 0.7523892267593397  0.7263249348392703  0.6950477845351868  0.6481320590790617  0.5838401390095569
             0.4813205907906168  0.3544743701129453  0.2033014769765421  0.07124239791485663 0.010425716768027803

"""

