# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
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
USE_IOU_FACTOR = False

VERSION = 'RetinaNet_DOTA_R3Det_2x_20191108'

"""
This is your result for task 1:

    mAP: 0.7066194189913816
    ap of each class:
    plane:0.8905480010393588,
    baseball-diamond:0.7845764249543027,
    bridge:0.4415489914209597,
    ground-track-field:0.6515721505439082,
    small-vehicle:0.7509226622459368,
    large-vehicle:0.7288453788151275,
    ship:0.8604046905135039,
    tennis-court:0.9082569687774237,
    basketball-court:0.8141347275878138,
    storage-tank:0.8253027715641935,
    soccer-ball-field:0.5623560181901192,
    roundabout:0.6100656068973895,
    harbor:0.5648618127447264,
    swimming-pool:0.6767393616949172,
    helicopter:0.5291557178810407

The submitted information is :

Description: RetinaNet_DOTA_R3Det_2x_20191108_70.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue


"""


