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
REG_LOSS_MODE = None

# CSL
LABEL_TYPE = 0  # {0: gaussian_label, 1: rectangular_label, 2: pulse_label, 3: triangle_label}
RADIUS = 1
OMEGA = 10

VERSION = 'RetinaNet_FDDB_CSL_2x_20211106'

"""
2007
cls : face|| Recall: 0.9765840220385675 || Precison: 0.2817405126167296|| AP: 0.9079883799827272
F1:0.9503847987199303 P:0.9788321167883212 R:0.9235537190082644
mAP is : 0.9079883799827272

2012
cls : face|| Recall: 0.9765840220385675 || Precison: 0.2817405126167296|| AP: 0.966448818204023
F1:0.9503847987199303 P:0.9788321167883212 R:0.9235537190082644
mAP is : 0.966448818204023

AP50:95
0.966448818204023  0.9507760766610001  0.9335603757163065  0.9057298188107186  0.8604708783393896
0.7322426626412546  0.4670438647856046  0.17136408579845444  0.017382884000010514  0.00014304018595622355


"""


