# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

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
REG_LOSS_MODE = None

VERSION = 'FCOS_HRSC2016_1x_20210616'

"""
FCOS + bs=2
FLOPs: 467903310;    Trainable params: 32057866

cls : ship|| Recall: 0.8517915309446255 || Precison: 0.5323155216284987|| AP: 0.7430047542210505
F1:0.7791283669347892 P:0.8059181897302001 R:0.754071661237785
mAP is : 0.7430047542210505

"""



