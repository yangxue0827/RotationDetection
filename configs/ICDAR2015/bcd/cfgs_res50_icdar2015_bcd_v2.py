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
SAVE_WEIGHTS_INTE = 10000
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

# loss
BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

# post-processing
VIS_SCORE = 0.78

VERSION = 'RetinaNet_ICDAR2015_BCD_2x_20210724'

"""
FLOPs: 472715460;    Trainable params: 32325246
0.78: Calculated!{"precision": 0.8413140311804009, "recall": 0.7274915743861339, "hmean": 0.7802736896462691, "AP": 0}%
0.77: Calculated!{"precision": 0.8373893805309734, "recall": 0.7289359653346172, "hmean": 0.7794079794079793, "AP": 0}%
0.75: Calculated!{"precision": 0.8234031132581857, "recall": 0.7385652383245065, "hmean": 0.7786802030456854, "AP": 0}%
0.8: Calculated!{"precision": 0.8506010303377218, "recall": 0.7154549831487723, "hmean": 0.7771966527196653, "AP": 0}%
0.85: Calculated!{"precision": 0.8705302096177558, "recall": 0.6798266730861819, "hmean": 0.7634495809678291, "AP": 0}%
0.83: Calculated!{"precision": 0.8609785202863962, "recall": 0.6947520462205103, "hmean": 0.7689848121502797, "AP": 0}%

Hmean50:95: 0.7802736896462691  0.7585850761683449  0.7250193648334624  0.6651174799896721  0.5830105861089594
            0.45442809191840955  0.2685256906790601  0.10534469403563129  0.01755744900593855  0.0005163955589981926

"""

