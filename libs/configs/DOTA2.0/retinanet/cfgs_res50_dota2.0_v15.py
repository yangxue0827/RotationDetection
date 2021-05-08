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
SAVE_WEIGHTS_INTE = 40000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

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

VERSION = 'RetinaNet_DOTA2.0_2x_20210504'

"""
retinanet-180
FLOPs: 865678480;    Trainable params: 33148131

This is your evaluation result for task 1:

    mAP: 0.43069494957862986
    ap of each class:
    plane:0.7511935607921781,
    baseball-diamond:0.47014450290932247,
    bridge:0.337831596208814,
    ground-track-field:0.5641908529585304,
    small-vehicle:0.3357549886530301,
    large-vehicle:0.3149136163232269,
    ship:0.4352201923571177,
    tennis-court:0.719417829945509,
    basketball-court:0.5321146014976036,
    storage-tank:0.49831325441562196,
    soccer-ball-field:0.40461147028865235,
    roundabout:0.49565498483846526,
    harbor:0.3477528948290374,
    swimming-pool:0.49707150306239245,
    helicopter:0.4399022806906489,
    container-crane:0.10935441370223979,
    airport:0.4230066364666493,
    helipad:0.07605991247629705

The submitted information is :

Description: RetinaNet_DOTA2.0_2x_20210504_104w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

