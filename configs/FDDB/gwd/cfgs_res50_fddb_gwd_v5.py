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
REG_LOSS_MODE = 2  # GWD loss

GWD_TAU = 2.0
GWD_FUNC = tf.sqrt   # 0: sqrt  1: log

VERSION = 'RetinaNet_FDDB_GWD_2x_20211107'

"""
RetinaNet-H + gwd
FLOPs: 830085208;    Trainable params: 32159286

2007
cls : face|| Recall: 0.9793388429752066 || Precison: 0.6970588235294117|| AP: 0.9084701338115604
F1:0.9562734202924636 P:0.9715707178393745 R:0.9414600550964187
mAP is : 0.9084701338115604

2012
cls : face|| Recall: 0.9793388429752066 || Precison: 0.6974006866110839|| AP: 0.9743959998654319
F1:0.9562734202924636 P:0.9715707178393745 R:0.9414600550964187
mAP is : 0.9743959998654319

AP50:95
0.9743959998654319  0.9683986266334633  0.9467748389119777  0.9140510627553924  0.879815780248925
0.8084355764595945  0.631735151961493  0.3637716489004802  0.08859798328279683  0.0011367744207063193
"""
