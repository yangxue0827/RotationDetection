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
SAVE_WEIGHTS_INTE = 40000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

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

VERSION = 'RetinaNet_DOTA2.0_CSL_2x_20210429'

"""
gaussian label, omega=10
FLOPs: 700124450;    Trainable params: 34019421
This is your evaluation result for task 1:

mAP: 0.43344110339222985
ap of each class:
plane:0.7474780683469242,
baseball-diamond:0.44284186301197104,
bridge:0.3602585746970695,
ground-track-field:0.5611689093593835,
small-vehicle:0.3430365248274545,
large-vehicle:0.32718842802965487, s
hip:0.4681204038689529,
tennis-court:0.7657646572876476,
basketball-court:0.5756647918918942,
storage-tank:0.4829006246589788,
soccer-ball-field:0.38423561115882415,
roundabout:0.4765156959674269,
harbor:0.3733914917262925,
swimming-pool:0.5137861147989039,
helicopter:0.38742437003275904,
container-crane:0.10810276679841897,
airport:0.42231789203218606,
helipad:0.061743072565393804

The submitted information is :

Description: RetinaNet_DOTA2.0_CSL_2x_20210429_104w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""


