# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 5.0

VERSION = 'RetinaNet_DOTA_R3Det_KF_2x_20210911'

"""
r3det + kfiou (exp(1-IoU)-1)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

FLOPs: 1032047562;    Trainable params: 37769656

This is your result for task 1:

    mAP: 0.7158259517675741
    ap of each class:
    plane:0.8939358923308309,
    baseball-diamond:0.758349832439058,
    bridge:0.45062227221169965,
    ground-track-field:0.6790243790298099,
    small-vehicle:0.7568629121004316,
    large-vehicle:0.7818254495705386,
    ship:0.8699980361387275,
    tennis-court:0.9071274512740564,
    basketball-court:0.831396396644877,
    storage-tank:0.806022073057049,
    soccer-ball-field:0.5715041910195074,
    roundabout:0.5972526434858508,
    harbor:0.6071834196574328,
    swimming-pool:0.6871565982317153,
    helicopter:0.5391277293220285

The submitted information is :

Description: RetinaNet_DOTA_R3Det_KF_2x_20210911_70.2w
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui
"""

