# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_DOTA_RIDet_2x_20210507'

"""
RIDet-8p
FLOPs: 1054755101;    Trainable params: 33148131

This is your result for task 1:

    mAP: 0.6605859069633909
    ap of each class:
    plane:0.8843655730283845,
    baseball-diamond:0.7701068450880688,
    bridge:0.4080972860665947,
    ground-track-field:0.6484822940328212,
    small-vehicle:0.6763128621076336,
    large-vehicle:0.5544593257763404,
    ship:0.7242105367854795,
    tennis-court:0.8953990214989828,
    basketball-court:0.7708644755434352,
    storage-tank:0.7808520730916176,
    soccer-ball-field:0.5277796924298762,
    roundabout:0.6474793137209606,
    harbor:0.5548762085711737,
    swimming-pool:0.604873184417449,
    helicopter:0.4606299122920471

The submitted information is :

Description: RetinaNet_DOTA_RIDet_2x_20210507_70.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
