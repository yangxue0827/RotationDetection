# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
from configs._base_.models.faster_rcnn_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 27000
DECAY_EPOCH = [24, 32, 40]
MAX_EPOCH = 34
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
# backbone
NET_NAME = 'resnet152_v1d'
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

# rpn sample
RPN_TOP_K_NMS_TEST = 12000
RPN_MAXIMUM_PROPOSAL_TEST = 3000

VERSION = 'FPN_Res152D_DOTA_BCD_2x_20210902'

"""
R2CNN + BCD
FLOPs: 1080337585;    Trainable params: 76310154

This is your result for task 1:

mAP: 0.7954123845615773
ap of each class: plane:0.903323080331905, baseball-diamond:0.8540217546181903, bridge:0.5933282757774304, ground-track-field:0.8114197923083153, small-vehicle:0.7908977794512878, large-vehicle:0.8265937021089517, ship:0.8706441486071012, tennis-court:0.9072727272727275, basketball-court:0.8786923968933851, storage-tank:0.8718289712846833, soccer-ball-field:0.7385516371615769, roundabout:0.6621865859365743, harbor:0.7670245409934106, swimming-pool:0.7205707017666689, helicopter:0.7348296739114518
The submitted information is :

Description: FPN_Res152D_DOTA_BCD_2x_20210902_91.8w_cpunms_mss_v1
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
