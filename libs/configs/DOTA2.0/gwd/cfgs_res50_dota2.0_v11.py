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
SAVE_WEIGHTS_INTE = 40000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

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

VERSION = 'RetinaNet_DOTA2.0_GWD_2x_20210318'

"""
RetinaNet-H + gwd fix bug + sqrt + tau=2
FLOPs: 865678525;    Trainable params: 33148131
This is your evaluation result for task 1:

mAP: 0.4664986415094291
ap of each class: 
plane:0.7464946714975453, 
baseball-diamond:0.47123212754977634, 
bridge:0.40119724576685906, 
ground-track-field:0.5812917981538129, 
small-vehicle:0.40670249870344394, 
large-vehicle:0.4046850003037289, 
ship:0.5428447606933438, 
tennis-court:0.7681349919524145, 
basketball-court:0.5755971232693727, 
storage-tank:0.5025165889989989, 
soccer-ball-field:0.4003468511908982, 
roundabout:0.5040208763284377, 
harbor:0.4180317727230398, 
swimming-pool:0.5443735811337471, 
helicopter:0.4937620765192604, 
container-crane:0.10539934448852856, 
airport:0.4237950621680601, 
helipad:0.10654917572845726

The submitted information is :

Description: RetinaNet_DOTA2.0_GWD_2x_20210318_112w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue


SWA

This is your evaluation result for task 1:

    mAP: 0.47634086829937744
    ap of each class:
    plane:0.7668082454617845,
    baseball-diamond:0.48418812812821305,
    bridge:0.4075901704186716,
    ground-track-field:0.5820167236252256,
    small-vehicle:0.4051345869003496,
    large-vehicle:0.4167099036188684,
    ship:0.5396401906183046,
    tennis-court:0.7650083921519408,
    basketball-court:0.5829846026735821,
    storage-tank:0.5034460965788641,
    soccer-ball-field:0.4233857143369985,
    roundabout:0.5072543667424922,
    harbor:0.43125132941971067,
    swimming-pool:0.5547004579756261,
    helicopter:0.4990813435158377,
    container-crane:0.11896443067478767,
    airport:0.4449535830679548,
    helipad:0.14101736347958232

The submitted information is :

Description: RetinaNet_DOTA2.0_GWD_2x_20210318_swa10
Username: yangxue
Institute: UCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue
"""

