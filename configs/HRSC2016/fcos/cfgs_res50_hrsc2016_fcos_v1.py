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

VERSION = 'FCOS_HRSC2016_1x_20210611'

"""
FCOS
FLOPs: 467903316;    Trainable params: 32057866

cls : ship|| Recall: 0.8249185667752443 || Precison: 0.531479538300105|| AP: 0.732974085854581
F1:0.7626777327619213 P:0.807832422586521 R:0.7223127035830619
mAP is : 0.732974085854581

cls : ship|| Recall: 0.3737785016286645 || Precison: 0.24081846799580273|| AP: 0.24628577652539932
F1:0.3755909511796513 P:0.4528735632183908 R:0.32084690553745926
mAP is : 0.24628577652539932


"""



