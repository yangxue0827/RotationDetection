# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2,3'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_EPOCH = [36, 48, 60]
MAX_EPOCH = 51
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
IMG_SHORT_SIDE_LEN = [800, 450, 500, 640, 700, 900, 1000, 1100, 1200]
IMG_MAX_LENGTH = 1200
# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = True

# model
NET_NAME = 'resnet152_v1d'
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

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA_R3Det_BCD_6x_20210816'

"""
r3det + bcd
FLOPs: 2264299626;    Trainable params: 72307128

This is your result for task 1:

mAP: 0.7976067414937239
ap of each class: plane:0.896975812537946, baseball-diamond:0.866150380193985, bridge:0.5577679647171538, ground-track-field:0.7858174511616975, small-vehicle:0.797304260326288, large-vehicle:0.8435979887581329, ship:0.8791839240036783, tennis-court:0.9085885548955026, basketball-court:0.8674270089065269, storage-tank:0.8692190301249617, soccer-ball-field:0.7309657157328202, roundabout:0.6995200065455824, harbor:0.7860175357715543, swimming-pool:0.7323330509904586, helicopter:0.7432324377395724
The submitted information is :

Description: RetinaNet_DOTA_R3Det_BCD_6x_20210816_275.4w_cpunms_mss
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui


This is your result for task 1:

mAP: 0.7994659852740972
ap of each class: plane:0.898204490566338, baseball-diamond:0.8624464904842577, bridge:0.5735721468775771, ground-track-field:0.791051108398076, small-vehicle:0.7944125140339489, large-vehicle:0.8485790223296026, ship:0.8788728637403803, tennis-court:0.9088054421875075, basketball-court:0.8669816559165882, storage-tank:0.8689495329187268, soccer-ball-field:0.7343703776380549, roundabout:0.6980220010043342, harbor:0.7882810423927771, swimming-pool:0.7380496017372202, helicopter:0.7413914888860714
The submitted information is :

Description: RetinaNet_DOTA_R3Det_BCD_6x_20210816_275.4w_swa12_cpunms_mss
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue


This is your result for task 1:

mAP: 0.7962795175364281
ap of each class: plane:0.8978889647538881, baseball-diamond:0.8575697155543572, bridge:0.5723063608320381, ground-track-field:0.7968202162987016, small-vehicle:0.7811594652552212, large-vehicle:0.8236514946015792, ship:0.8738530715689999, tennis-court:0.9047498278805575, basketball-court:0.8570998016596487, storage-tank:0.8659052090118392, soccer-ball-field:0.7327477104260606, roundabout:0.688138328998597, harbor:0.781093057628145, swimming-pool:0.75013159093336, helicopter:0.7610779476434267
The submitted information is :

Description: RetinaNet_DOTA_R3Det_BCD_6x_20210816_275.4w_cpunms_mss_ms
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

"""

