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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_FDDB_2x_20211106'

"""
RetinaNet-H + 90
FLOPs: 830085163;    Trainable params: 32159286

2007
cls : face|| Recall: 0.9648760330578512 || Precison: 0.5751231527093597|| AP: 0.9071560203590661
F1:0.9482526582400714 P:0.9697624190064795 R:0.9276859504132231
mAP is : 0.9071560203590661

2012
cls : face|| Recall: 0.9648760330578512 || Precison: 0.574887156339762|| AP: 0.959204678220418
F1:0.9482526582400714 P:0.9697624190064795 R:0.9276859504132231
mAP is : 0.959204678220418

AP50:95
0.959204678220418  0.9301560772935049  0.8749958747257098  0.7844197465233099  0.683315839522552
0.558135300551797  0.3479441339258663  0.12669957890041392  0.011630901808271605  3.2424916862513164e-05

"""

