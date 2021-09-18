# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2,3'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 40000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

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

VERSION = 'RetinaNet_DOTA2.0_R3Det_KL_2x_20210426'

"""
r3det + kl + sqrt tau=2
FLOPs: 1269557700;    Trainable params: 37921786

This is your evaluation result for task 1:

    mAP: 0.5090209422835589
    ap of each class:
    plane:0.7912794411070341,
    baseball-diamond:0.4695487242250586,
    bridge:0.41533160379561873,
    ground-track-field:0.5677430487057211,
    small-vehicle:0.5356205901820142,
    large-vehicle:0.5584680505670461,
    ship:0.657336050883738,
    tennis-court:0.7772409744994606,
    basketball-court:0.5984195581564126,
    storage-tank:0.6494453557810127,
    soccer-ball-field:0.4395169657802019,
    roundabout:0.5123080948586409,
    harbor:0.456294580056635,
    swimming-pool:0.5760765572180435,
    helicopter:0.50117159491717,
    container-crane:0.1909383688656195,
    airport:0.4098954789380368,
    helipad:0.05574192256659425

The submitted information is :

Description: RetinaNet_DOTA2.0_R3Det_KL_2x_20210426_104w
Username: DetectionTeamCSU
Institute: UCAS
Emailadress: yangxue@csu.edu.cn
TeamMembers: yangxue
"""

