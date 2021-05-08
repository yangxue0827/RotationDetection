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

GWD_TAU = 2.0
GWD_FUNC = tf.sqrt

VERSION = 'RetinaNet_DOTA1.5_R3Det_GWD_2x_20210423'

"""
r3det+gwd (only refine stage) + sqrt tau=2
FLOPs: 1264993896;    Trainable params: 37820366

This is your evaluation result for task 1:

    mAP: 0.6322484937264011
    ap of each class:
    plane:0.8024987716942871,
    baseball-diamond:0.7425775413636828,
    bridge:0.4655616667665008,
    ground-track-field:0.6388393036437524,
    small-vehicle:0.5664905772922714,
    large-vehicle:0.7362379178484698,
    ship:0.8232844175002079,
    tennis-court:0.8994145813875651,
    basketball-court:0.7427083099769697,
    storage-tank:0.704031654812041,
    soccer-ball-field:0.47574388581472915,
    roundabout:0.6888700728764863,
    harbor:0.5986676035523161,
    swimming-pool:0.6501337037910943,
    helicopter:0.4678937782799318,
    container-crane:0.11302211302211303

The submitted information is :

Description: RetinaNet_DOTA1.5_R3Det_GWD_2x_20210423_102.w.zip
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

