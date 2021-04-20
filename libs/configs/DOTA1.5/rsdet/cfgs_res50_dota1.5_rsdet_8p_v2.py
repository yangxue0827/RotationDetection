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
REG_WEIGHT = 1.0

# post-processing
VIS_SCORE = 0.8

VERSION = 'RetinaNet_DOTA1.5_RSDet_2x_20210417'

"""
RSDet-8p
FLOPs: 677908676;    Trainable params: 33196536
This is your evaluation result for task 1:

    mAP: 0.6141596863067104
    ap of each class:
    plane:0.7926315950676603,
    baseball-diamond:0.7967054895751493,
    bridge:0.4160636156278779,
    ground-track-field:0.6703192155607761,
    small-vehicle:0.48422370847514784,
    large-vehicle:0.5340846752056462,
    ship:0.7355031533688142,
    tennis-court:0.8930360680372161,
    basketball-court:0.7508410392054757,
    storage-tank:0.6341013296442767,
    soccer-ball-field:0.5065672918094257,
    roundabout:0.6849936131749105,
    harbor:0.6196292257865295,
    swimming-pool:0.6427293050694697,
    helicopter:0.5507740648604437,
    container-crane:0.11435159043854697

The submitted information is :

Description: RetinaNet_DOTA1.5_RSDet_2x_20210417_108.8w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""


