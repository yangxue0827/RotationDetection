# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
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

# bbox head
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA_1x_20210725'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta)) + 180, sin^2(theta) + cos^2(theta) = 1
[-90, 90]   sin in [-1, 1]   cos in [0, 1]
FLOPs: 485784881;    Trainable params: 33051321
This is your result for task 1:

mAP: 0.6482820239385153
ap of each class: plane:0.8863486082518542, baseball-diamond:0.7510916490271552, bridge:0.4136498976633022, ground-track-field:0.6934357734426206, small-vehicle:0.5915433817529869, large-vehicle:0.4156886089040786, ship:0.6512479280213479, tennis-court:0.8965927064782218, basketball-court:0.778541563411186, storage-tank:0.7716242837257139, soccer-ball-field:0.5261143148330104, roundabout:0.6328490142731126, harbor:0.5072934651888339, swimming-pool:0.6566747539350666, helicopter:0.55153441016924
The submitted information is :

Description: RetinaNet_DOTA_1x_20210725_35.1w_v1
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



