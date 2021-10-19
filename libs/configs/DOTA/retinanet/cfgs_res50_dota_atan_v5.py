# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 27000 * 2
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

VERSION = 'RetinaNet_DOTA_2x_20210727'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta)) + 180, sin^2(theta) + cos^2(theta) = 1
[-90, 90]   sin in [-1, 1]   cos in [0, 1]
FLOPs: 862200848;    Trainable params: 33051321

This is your result for task 1:

mAP: 0.6532556290645889
ap of each class: plane:0.8879437201698563, baseball-diamond:0.7424327258991498, bridge:0.39876196060550234, ground-track-field:0.6762748480559626, small-vehicle:0.6215548266972836, large-vehicle:0.4303887925448767, ship:0.6997482727279865, tennis-court:0.8981939964160642, basketball-court:0.8132157603668859, storage-tank:0.7918250070749778, soccer-ball-field:0.532450059662894, roundabout:0.6126948545190352, harbor:0.5047173472183413, swimming-pool:0.651816449043771, helicopter:0.5368158149662463
The submitted information is :

Description: RetinaNet_DOTA_2x_20210727_70.2w_v1
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



