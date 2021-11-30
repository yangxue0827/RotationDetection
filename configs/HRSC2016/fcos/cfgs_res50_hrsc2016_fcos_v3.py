# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 2
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3 * BATCH_SIZE * NUM_GPU
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
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
CTR_WEIGHT = 1.0
REG_LOSS_MODE = 0

VERSION = 'FCOS_HRSC2016_1x_20210616_v2'

"""
FCOS + bs=2 + modulated loss
FLOPs: 467903316;    Trainable params: 32057866

cls : ship|| Recall: 0.8355048859934854 || Precison: 0.5674778761061947|| AP: 0.7504115977249796
F1:0.7851612514334804 P:0.8237924865831843 R:0.75
mAP is : 0.7504115977249796
"""



