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

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

# post-processing
VIS_SCORE = 0.99

VERSION = 'RetinaNet_ICDAR2015_BCD_1x_20210724'

"""
FLOPs: 472715456;    Trainable params: 32325246

0.99: Calculated!{"precision": 0.8327683615819209, "recall": 0.7096774193548387, "hmean": 0.7663114114894723, "AP": 0}%
0.995: Calculated!{"precision": 0.8508982035928143, "recall": 0.6841598459316321, "hmean": 0.758473445423005, "AP": 0}%
0.985: Calculated!{"precision": 0.815401419989077, "recall": 0.7188252286952335, "hmean": 0.7640736949846467, "AP": 0}%
0.98: Calculated!{"precision": 0.8052294557097118, "recall": 0.7265286470871449, "hmean": 0.7638572513287776, "AP": 0}%
0.97: Calculated!{"precision": 0.7930497925311203, "recall": 0.7361579200770342, "hmean": 0.7635455680399501, "AP": 0}%
0.96: Calculated!{"precision": 0.7805247225025227, "recall": 0.7448242657679345, "hmean": 0.7622567134762257, "AP": 0}%
0.95: Calculated!{"precision": 0.7713580246913581, "recall": 0.7520462205103514, "hmean": 0.7615797172111166, "AP": 0}%
0.94: Calculated!{"precision": 0.7642474427666829, "recall": 0.7554164660568127, "hmean": 0.7598062953995157, "AP": 0}%
0.93: Calculated!{"precision": 0.75563009103977, "recall": 0.7592681752527685, "hmean": 0.7574447646493757, "AP": 0}%
0.92: Calculated!{"precision": 0.75, "recall": 0.7626384207992296, "hmean": 0.7562664120315111, "AP": 0}%
0.9: Calculated!{"precision": 0.7381835032437442, "recall": 0.7669715936446798, "hmean": 0.7523022432113341, "AP": 0}
0.85: Calculated!{"precision": 0.7202514593623709, "recall": 0.772267693789119, "hmean": 0.7453531598513011, "AP": 0}%

Hmean50:95: 0.7663114114894723  0.7455159864829738  0.7106836495970886  0.6644138289576293  0.5682349883025734
            0.43098518325968294 0.2630621263322069  0.10241746815700545 0.024954510007798282 0.0015596568754873926

"""

