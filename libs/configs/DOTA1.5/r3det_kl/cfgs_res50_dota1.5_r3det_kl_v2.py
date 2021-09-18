# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 32000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

KL_TAU = 2.0
KL_FUNC = 1   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA1.5_R3Det_KL_2x_20210423'

"""
r3det + kl + sqrt tau=2
FLOPs: 1033867194;    Trainable params: 37820366

This is your evaluation result for task 1:

    mAP: 0.6517834978722121
    ap of each class:
    plane:0.8029868768935169,
    baseball-diamond:0.7274977717100799,
    bridge:0.4737323854654457,
    ground-track-field:0.6018401719491528,
    small-vehicle:0.6324111068964658,
    large-vehicle:0.7514635107769393,
    ship:0.8607054711211418,
    tennis-court:0.8952458550652271,
    basketball-court:0.7352291137148709,
    storage-tank:0.7292558330841545,
    soccer-ball-field:0.5035864952531544,
    roundabout:0.6623234599022446,
    harbor:0.6457653133001389,
    swimming-pool:0.6910883017090202,
    helicopter:0.5783718622915984,
    container-crane:0.13703243682224242

The submitted information is :

Description: RetinaNet_DOTA1.5_R3Det_KL_2x_20210423_108.8w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

