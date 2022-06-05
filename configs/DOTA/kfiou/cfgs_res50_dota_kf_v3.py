# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 27000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_DOTA_KF_1x_20210902'

"""
RetinaNet-H + kfiou (exp(1-IoU)-1)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

This is your evaluation result for task 1:

    mAP: 0.7068630294977952
    ap of each class: 
    plane:0.894332195251659, 
    baseball-diamond:0.7788464461771812, 
    bridge:0.4539694438214467, 
    ground-track-field:0.6706240268477037, 
    small-vehicle:0.7246726968745004, 
    large-vehicle:0.6687244421277758, 
    ship:0.7856459435671667, 
    tennis-court:0.9080137077942301, 
    basketball-court:0.8307342665474495, 
    storage-tank:0.7912333630665858, 
    soccer-ball-field:0.5858193181559574, 
    roundabout:0.6591340351162999, 
    harbor:0.6350698669491817, 
    swimming-pool:0.6848033517289643, 
    helicopter:0.5313223384408268

The submitted information is :

Description: RetinaNet_DOTA_KF_1x_20210902
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



