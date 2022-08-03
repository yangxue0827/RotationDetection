# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
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
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_REFINE_STAGE = 1
NUM_SUBNET_CONV = 5

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA_R3Det_BCD_6x_20210823'

"""
r3det + bcd
FLOPs: 1117012922;    Trainable params: 40129976

This is your result for task 1:

mAP: 0.7762858405073361
ap of each class: plane:0.8953497226547358, baseball-diamond:0.8501656824907948, bridge:0.5431991929442357, ground-track-field:0.7307740576742793, small-vehicle:0.7910599417201704, large-vehicle:0.8483070490452013, ship:0.8805073193004167, tennis-court:0.9082058455052475, basketball-court:0.8592469779916725, storage-tank:0.8574868800443416, soccer-ball-field:0.6680115081809131, roundabout:0.6396178803360391, harbor:0.7710649675417125, swimming-pool:0.7381719110769228, helicopter:0.6631186711033582
The submitted information is :

Description: RetinaNet_DOTA_R3Det_BCD_6x_20210823_275.4w_cpunms
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui


This is your result for task 1:

mAP: 0.7908035910190879
ap of each class: plane:0.8976703143322362, baseball-diamond:0.8611070242387947, bridge:0.564765660971753, ground-track-field:0.8040901005452837, small-vehicle:0.7936863490371714, large-vehicle:0.8503420736648556, ship:0.881463078396147, tennis-court:0.9083051633849253, basketball-court:0.8676716053481768, storage-tank:0.8690471930544194, soccer-ball-field:0.7187335107838447, roundabout:0.6332232774174681, harbor:0.7808444069017951, swimming-pool:0.7431050852866289, helicopter:0.6879990219228187
The submitted information is :

Description: RetinaNet_DOTA_R3Det_BCD_6x_20210823_275.4w_cpunms_mss
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui


This is your result for task 1:

mAP: 0.7889208745161991
ap of each class: plane:0.8975101844165517, baseball-diamond:0.8510492079566572, bridge:0.5602981576062379, ground-track-field:0.8193540594321004, small-vehicle:0.7856903990701805, large-vehicle:0.8413358516289686, ship:0.8770097880713499, tennis-court:0.9081521172162191, basketball-court:0.8658819992122361, storage-tank:0.8704102746915239, soccer-ball-field:0.704359555378958, roundabout:0.666733137127285, harbor:0.778503052286466, swimming-pool:0.7332044618102787, helicopter:0.6743208718379766
The submitted information is :

Description: RetinaNet_DOTA_R3Det_BCD_6x_20210823_275.4w_swa11_cpunms_mss
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

