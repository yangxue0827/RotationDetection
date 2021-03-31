# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 27000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# neck
FPN_MODE = 'scrdet'

# rpn head
USE_CENTER_OFFSET = False
BASE_ANCHOR_SIZE_LIST = 256
ANCHOR_STRIDE = 8
ANCHOR_SCALES = [0.0625, 0.125, 0.25, 0.5, 1., 2.0]
ANCHOR_RATIOS = [0.5, 1., 2.0, 1/4.0, 4.0, 1/6.0, 6.0]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0, 10.0]
ANCHOR_SCALE_FACTORS = None
ANCHOR_MODE = 'H'
ANGLE_RANGE = 90

# loss
USE_IOU_FACTOR = False

# post-processing
FAST_RCNN_H_NMS_IOU_THRESHOLD = 0.3
FAST_RCNN_R_NMS_IOU_THRESHOLD = 0.2

VERSION = 'FPN_Res50D_DOTA_1x_20201103'

"""
SCRDet
FLOPs: 1126555840;    Trainable params: 43227534

This is your result for task 1:

mAP: 0.7017537811932414
ap of each class: plane:0.8911387826735638, baseball-diamond:0.775529003353518, bridge:0.4509789898495809, ground-track-field:0.6496553432174655, small-vehicle:0.753619348187285, large-vehicle:0.7440730940099484, ship:0.8512441917843704, tennis-court:0.898862230577996, basketball-court:0.8331792736551014, storage-tank:0.8204919677785293, soccer-ball-field:0.42899647699961085, roundabout:0.6034139167456652, harbor:0.6458833428224676, swimming-pool:0.6753589974983456, helicopter:0.5038817587451719
The submitted information is :

Description: FPN_Res50D_DOTA_1x_20201103_37.8w_r
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

This is your evaluation result for task 2:

mAP: 0.7172708030045355
ap of each class: plane:0.8948601687941339, baseball-diamond:0.7983222545157133, bridge:0.5043454171429804, ground-track-field:0.6251083846010812, small-vehicle:0.7672174612590384, large-vehicle:0.7431535330369287, ship:0.8542223630049876, tennis-court:0.9086082885400426, basketball-court:0.8419754132368138, storage-tank:0.825103739540167, soccer-ball-field:0.4284043211050202, roundabout:0.601584793813972, harbor:0.7417571712916949, swimming-pool:0.7062648105990572, helicopter:0.5181339245864024
The submitted information is :

Description: FPN_Res50D_DOTA_1x_20201103_37.8w_h
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

