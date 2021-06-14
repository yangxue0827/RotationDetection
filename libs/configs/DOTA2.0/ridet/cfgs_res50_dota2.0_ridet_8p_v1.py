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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_DOTA2.0_RIDet_2x_20210509'

"""
RIDet-8p
FLOPs: 870905839;    Trainable params: 33293346

This is your evaluation result for task 1:

    mAP: 0.45350021903066295
    ap of each class:
    plane:0.7640386782607965,
    baseball-diamond:0.45941949819838707,
    bridge:0.3822849941827053,
    ground-track-field:0.5882485982911329,
    small-vehicle:0.38182163088601057,
    large-vehicle:0.35913778649186495,
    ship:0.4927984243614078,
    tennis-court:0.7627193407094273,
    basketball-court:0.541613446379996,
    storage-tank:0.521356661130177,
    soccer-ball-field:0.41027757525094277,
    roundabout:0.5003811125017407,
    harbor:0.41894648024188597,
    swimming-pool:0.5081730389824448,
    helicopter:0.4385247995863651,
    container-crane:0.058974774482795866,
    airport:0.5203210333539394,
    helipad:0.05396606925991192

The submitted information is :

Description: RetinaNet_DOTA2.0_RIDet_2x_20210509_104w
Username: yangxue
Institute: UCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue
"""
