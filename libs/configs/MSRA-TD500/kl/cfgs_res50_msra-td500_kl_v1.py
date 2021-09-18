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
REG_LOSS_MODE = 3  # KLD loss

KL_TAU = 2.0
KL_FUNC = 0   # 0: sqrt  1: log

# post-processing
VIS_SCORE = 0.75

VERSION = 'RetinaNet_MSRA_TD500_KL_2x_20210215'

"""
FLOPs: 472715463;    Trainable params: 32325246
0.75: Calculated!{"precision": 0.7912885662431942, "recall": 0.7491408934707904, "hmean": 0.7696381288614298, "AP": 0}
0.70: Calculated!{"precision": 0.7849462365591398, "recall": 0.7525773195876289, "hmean": 0.768421052631579, "AP": 0}
0.80: Calculated!{"precision": 0.7977736549165121, "recall": 0.738831615120275, "hmean": 0.7671721677074043, "AP": 0}

Hmean50:95 = 0.7696381288614298 0.7396293027360988 0.7007943512797882 0.6601941747572815 0.5913503971756399
             0.46954986760812006 0.33362753751103263 0.19593998234774934 0.061782877316857894 0.00176522506619594
0.45242718446601937
"""

