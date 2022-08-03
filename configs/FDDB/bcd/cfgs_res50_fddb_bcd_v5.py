# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

# eval
USE_07_METRIC = False

VERSION = 'RetinaNet_FDDB_BCD_2x_20211107'

"""
RetinaNet-H + bcd
FLOPs: 830085199;    Trainable params: 32159286

2007
cls : face|| Recall: 0.9731404958677686 || Precison: 0.552168815943728|| AP: 0.9080855358707782
F1:0.9572539224605103 P:0.9825960841189267 R:0.9331955922865014
mAP is : 0.9080855358707782

2012
cls : face|| Recall: 0.9731404958677686 || Precison: 0.552168815943728|| AP: 0.9667116088409442
F1:0.9572539224605103 P:0.9825960841189267 R:0.9331955922865014
mAP is : 0.9667116088409442

AP50:95
0.9667116088409442  0.9583747367120262  0.9459712171068818  0.9187887651179382  0.8870967751940114
0.830905213964575  0.683485676106417  0.40715324760975835  0.1035741735584503

"""
