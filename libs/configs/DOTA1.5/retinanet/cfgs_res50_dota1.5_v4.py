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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA1.5_2x_20210314'

"""
RetinaNet-H + 90
FLOPs: 862193617;    Trainable params: 33051321
This is your evaluation result for task 1:

mAP: 0.5887471058289246
ap of each class:
plane:0.7849039138235574,
baseball-diamond:0.7217680640990516,
bridge:0.394302700294054,
ground-track-field:0.622515144946226,
small-vehicle:0.47130444935356086,
large-vehicle:0.4864076567136256,
ship:0.7364028142317903,
tennis-court:0.8965332860918587,
basketball-court:0.7527122732520811,
storage-tank:0.5898721957198321,
soccer-ball-field:0.49463409572934963,
roundabout:0.6811650594270172,
harbor:0.5194446376124553,
swimming-pool:0.6394792338041666,
helicopter:0.5323204848797101,
container-crane:0.09618768328445748

The submitted information is :

Description: RetinaNet_DOTA1.5_2x_20210314_83.2w
Username: AICyber
Institute: IECAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: Yang Xue; Yang Jirui
"""
