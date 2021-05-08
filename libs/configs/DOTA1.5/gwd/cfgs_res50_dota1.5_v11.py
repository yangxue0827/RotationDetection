# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 32000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0
REG_LOSS_MODE = 2  # GWD loss

GWD_TAU = 2.0
GWD_FUNC = tf.sqrt

VERSION = 'RetinaNet_DOTA1.5_GWD_2x_20210320'

"""
RetinaNet-H + gwd fix bug + sqrt + tau=2
FLOPs: 862193665;    Trainable params: 33051321
This is your evaluation result for task 1:

    mAP: 0.600346226078195
    ap of each class:
    plane:0.7939431922975534,
    baseball-diamond:0.7446174663646932,
    bridge:0.4195239451373103,
    ground-track-field:0.6015244842574132,
    small-vehicle:0.4997726370365391,
    large-vehicle:0.6002761372691583,
    ship:0.7616626473950818,
    tennis-court:0.9031303299213864,
    basketball-court:0.7175429727485252,
    storage-tank:0.5848017376755156,
    soccer-ball-field:0.48043525038685025,
    roundabout:0.6779911622252887,
    harbor:0.5579857368665273,
    swimming-pool:0.6579145411010502,
    helicopter:0.5135082856591359,
    container-crane:0.09090909090909091

The submitted information is :

Description: RetinaNet_DOTA1.5_GWD_2x_20210320_83.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue


SWA

This is your evaluation result for task 1:

    mAP: 0.6060358478025953
    ap of each class:
    plane:0.7967255029000768,
    baseball-diamond:0.7429314386766407,
    bridge:0.4386606848687486,
    ground-track-field:0.61637498980047,
    small-vehicle:0.4954853728442736,
    large-vehicle:0.5942575317263504,
    ship:0.7612793329714422,
    tennis-court:0.8943097462172913,
    basketball-court:0.7477277785488338,
    storage-tank:0.591234002230469,
    soccer-ball-field:0.4684316925081105,
    roundabout:0.6707910130275366,
    harbor:0.5665038559497408,
    swimming-pool:0.6698656284482921,
    helicopter:0.5484466070264743,
    container-crane:0.0935483870967742

The submitted information is :

Description: RetinaNet_DOTA1.5_GWD_2x_20210320_swa12
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

