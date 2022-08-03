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
SAVE_WEIGHTS_INTE = 32000
DECAY_EPOCH = [24, 32, 40]
MAX_EPOCH = 34
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16
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

VERSION = 'FPN_Res152D_DOTA1.5_BCD_2x_20211023'

"""
R2CNN + BCD


This is your evaluation result for task 1:

mAP: 0.732609593824819
ap of each class: plane:0.8095247359293846, baseball-diamond:0.8448784759256102, bridge:0.5543783104213547, ground-track-field:0.7759074900025141, small-vehicle:0.6592290259851553, large-vehicle:0.7803342794460464, ship:0.887470342413165, tennis-court:0.9082128678152768, basketball-court:0.8537443158434966, storage-tank:0.8373206056855428, soccer-ball-field:0.6833640934391939, roundabout:0.7493892400262232, harbor:0.7349858492083546, swimming-pool:0.7476870437870514, helicopter:0.7790501007830554, container-crane:0.11627672448567972
The submitted information is :

Description: FPN_Res152D_DOTA1.5_BCD_2x_20211023_108.8w_mss
"""
