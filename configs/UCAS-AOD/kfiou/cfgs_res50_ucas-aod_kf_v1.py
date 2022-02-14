# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

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

# test and eval
USE_07_METRIC = False

VERSION = 'RetinaNet_UCAS-AOD_KF_2x_20210818'

"""

USE_07_METRIC
cls : plane|| Recall: 0.9900677200902934 || Precison: 0.5736332722992414|| AP: 0.902354395930606
F1:0.9728333884736308 P:0.9754879709487063 R:0.9702031602708804
cls : car|| Recall: 0.977302204928664 || Precison: 0.2974733517568101|| AP: 0.8853851598308646
F1:0.9090859104416241 P:0.8945386064030132 R:0.9241245136186771
mAP is : 0.8938697778807353

USE_12_METRIC
cls : car|| Recall: 0.977302204928664 || Precison: 0.2974733517568101|| AP: 0.9421959167327536
F1:0.9090859104416241 P:0.8945386064030132 R:0.9241245136186771
cls : plane|| Recall: 0.9900677200902934 || Precison: 0.5736332722992414|| AP: 0.9803241564620929
F1:0.9728333884736308 P:0.9754879709487063 R:0.9702031602708804
mAP is : 0.9612600365974233
"""
