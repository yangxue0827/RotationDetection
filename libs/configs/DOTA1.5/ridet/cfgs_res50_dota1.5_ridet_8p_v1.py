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
LR = 1e-3
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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_DOTA1.5_RIDet_2x_20210507'

"""
RIDet-8p
FLOPs: 867420979;    Trainable params: 33196536

This is your evaluation result for task 1:

    mAP: 0.5891065123755762
    ap of each class:
    plane:0.7886888950536876,
    baseball-diamond:0.7405809701143095,
    bridge:0.4088215843507053,
    ground-track-field:0.65610145187512,
    small-vehicle:0.4795821143757697,
    large-vehicle:0.5486902794385126,
    ship:0.7205878393400889,
    tennis-court:0.894340892128819,
    basketball-court:0.7411365282482376,
    storage-tank:0.6153435436544774,
    soccer-ball-field:0.473378253519403,
    roundabout:0.6823304292827477,
    harbor:0.5872830734569421,
    swimming-pool:0.6268971841888457,
    helicopter:0.4611858908582639,
    container-crane:0.0007552681232899498

The submitted information is :

Description: RetinaNet_DOTA1.5_RIDet_2x_20210507_83.2w
Username: AICyber
Institute: IECAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: Yang Xue; Yang Jirui
"""
