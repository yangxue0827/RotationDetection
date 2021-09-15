# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 32000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 2.0
REG_LOSS_MODE = None

# CSL
LABEL_TYPE = 0  # {0: gaussian_label, 1: rectangular_label, 2: pulse_label, 3: triangle_label}
RADIUS = 1
OMEGA = 10

VERSION = 'RetinaNet_DOTA1.5_CSL_2x_20210414'

"""
gaussian label, omega=10
FLOPs: 697510828;    Trainable params: 33922611
This is your evaluation result for task 1:

    mAP: 0.5854712426655257
    ap of each class:
    plane:0.7827784330552927,
    baseball-diamond:0.7515107165309902,
    bridge:0.40132452946731767,
    ground-track-field:0.6102084932562825,
    small-vehicle:0.46426904540468233,
    large-vehicle:0.5084532143309923,
    ship:0.7259965936578786,
    tennis-court:0.8980010321312986,
    basketball-court:0.735872460903648,
    storage-tank:0.6046482059720273,
    soccer-ball-field:0.5145205345029095,
    roundabout:0.6408216451970133,
    harbor:0.536169920762916,
    swimming-pool:0.6419707154258086,
    helicopter:0.45957851150308926,
    container-crane:0.09141583054626533

The submitted information is :

Description: RetinaNet_DOTA1.5_CSL_2x_20210414_83.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""


