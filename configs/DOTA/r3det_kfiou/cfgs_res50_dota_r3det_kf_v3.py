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
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_SUBNET_CONV = 5
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 5.0

VERSION = 'RetinaNet_DOTA_R3Det_KF_6x_20210915'

"""
r3det + kfiou -ln(IoU)
FLOPs: 1368930204;    Trainable params: 40129976

This is your result for task 1:

mAP: 0.7669648012396494
ap of each class: plane:0.8903598876856382, baseball-diamond:0.8403556786718249, bridge:0.5298106107081239, ground-track-field:0.7300052181951444, small-vehicle:0.7869067536414414, large-vehicle:0.8360010396508978, ship:0.8760996973617721, tennis-court:0.9079439581536293, basketball-court:0.8596746226632423, storage-tank:0.8547449698669649, soccer-ball-field:0.6477300081741808, roundabout:0.6329207266210739, harbor:0.6918442269671817, swimming-pool:0.7638234411004695, helicopter:0.6562511791331542
The submitted information is :

Description: RetinaNet_DOTA_R3Det_KF_6x_20210915_swa12
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

mAP: 0.782983632347527
ap of each class: plane:0.8955533003377223, baseball-diamond:0.8476497452550542, bridge:0.5530871610734395, ground-track-field:0.8239142315558526, small-vehicle:0.7886623836054383, large-vehicle:0.8394577060879312, ship:0.8799454988304086, tennis-court:0.907790847506361, basketball-court:0.8586019766290067, storage-tank:0.857681422223366, soccer-ball-field:0.634637341417353, roundabout:0.6343882009106873, harbor:0.767125932224765, swimming-pool:0.7911007728323555, helicopter:0.6651579647231655
The submitted information is :

Description: RetinaNet_DOTA_R3Det_KF_6x_20210915_swa12_mss
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue
"""

