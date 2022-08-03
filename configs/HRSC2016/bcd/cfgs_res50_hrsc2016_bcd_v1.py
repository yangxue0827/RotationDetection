# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = "2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 2500
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'HRSC2016'
IMG_SHORT_SIDE_LEN = 512
IMG_MAX_LENGTH = 512
CLASS_NUM = 1

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_HRSC2016_BCD_1x_20210719'

"""
RetinaNet-H + bcd

1-1/(sqrt(bcd)+2)
cls : ship|| Recall: 0.9389250814332247 || Precison: 0.33891828336272783|| AP: 0.8006910838330422
F1:0.8481212376269202 P:0.8224942616679418 R:0.8754071661237784
mAP is : 0.8006910838330422

sqrt(bcd)
cls : ship|| Recall: 0.9438110749185668 || Precison: 0.25422241719675365|| AP: 0.8194797407475368
F1:0.8538472667619844 P:0.8333333333333334 R:0.8754071661237784
mAP is : 0.8194797407475368

log(bcd)
cls : ship|| Recall: 0.9535830618892508 || Precison: 0.2864481409001957|| AP: 0.8067709681301083
F1:0.8423027076744687 P:0.7981049562682215 R:0.8916938110749185
mAP is : 0.8067709681301083
"""


