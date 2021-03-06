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
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA1.5_2x_20210503'

"""
retinanet-180
FLOPs: 862193566;    Trainable params: 33051321
This is your evaluation result for task 1:

mAP: 0.39067994750968604
ap of each class: plane:0.6971214547040069, baseball-diamond:0.6718743603828817, bridge:0.2070197253659081, ground-track-field:0.40821796977456143, small-vehicle:0.19107489374332567, large-vehicle:0.23599989376588054, ship:0.31153479348200586, tennis-court:0.57307407579501, basketball-court:0.5579505327740116, storage-tank:0.46645234537021335, soccer-ball-field:0.3431638970348647, roundabout:0.46736744088552895, harbor:0.2873997032766335, swimming-pool:0.32567754830545365, helicopter:0.5029979563247304, container-crane:0.003952569169960474
The submitted information is :

Description: RetinaNet_DOTA1.5_2x_20210503_83.2w
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: yangxue
"""

