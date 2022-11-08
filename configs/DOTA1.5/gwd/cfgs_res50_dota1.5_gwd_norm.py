# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

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
REG_WEIGHT = 3.0
REG_LOSS_MODE = 5  # GWD norm loss

VERSION = 'RetinaNet_DOTA1.5_GWD_NORM_1x_20220628'

"""
RetinaNet-H + gwd norm
FLOPs: 862199428;    Trainable params: 33051321
This is your evaluation result for task 1 (VOC metrics):

    mAP: 0.6239493322511103
    ap of each class: 
    plane:0.7978956445978564, 
    baseball-diamond:0.7778363035922893, 
    bridge:0.4314213704046725, 
    ground-track-field:0.6718337794592112, 
    small-vehicle:0.4986054492458443, 
    large-vehicle:0.6322865940479369, 
    ship:0.7779814571760638, 
    tennis-court:0.9081527271046475, 
    basketball-court:0.7668087809344389, 
    storage-tank:0.591621743545823, 
    soccer-ball-field:0.5021806263185661, 
    roundabout:0.6773311121873372, 
    harbor:0.6302059156938736, 
    swimming-pool:0.651394852733292, 
    helicopter:0.5539965953395476, 
    container-crane:0.11363636363636365

COCO style result:

AP50: 0.6239493322511103
AP75: 0.3763800057230732
mAP: 0.3690782293575766
The submitted information is :

Description: RetinaNet_DOTA1.5_GWD_NORM_1x_20220628_83.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
