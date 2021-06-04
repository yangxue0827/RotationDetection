# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1"
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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA2.0_RSDet_2x_20210518'

"""
RSDet-5p
FLOPs: 676601855;    Trainable params: 33148131
This is your evaluation result for task 1:

    mAP: 0.4517227931303597
    ap of each class:
    plane:0.751246710552927,
    aseball-diamond:0.4684318375003039,
    bridge:0.37582986461134094,
    ground-track-field:0.5874242808519564,
    small-vehicle:0.34556213561144244,
    large-vehicle:0.34684288083603637,
    ship:0.4657224377370115,
    tennis-court:0.7654731223227706,
    basketball-court:0.5729850525336201,
    storage-tank:0.502777521324929,
    occer-ball-field:0.41512987195545975,
    roundabout:0.48511810153440377,
    harbor:0.36914552610446694,
    swimming-pool:0.5334466751996363,
    helicopter:0.45742116085027446,
    container-crane:0.11350649350649351,
    airport:0.45714359714879793,
    helipad:0.11780300616460358

The submitted information is :

Description: RetinaNet_DOTA2.0_RSDet_2x_20210518_104w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""


