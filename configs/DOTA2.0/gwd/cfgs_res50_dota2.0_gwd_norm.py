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
REG_WEIGHT = 3.0
REG_LOSS_MODE = 5  # GWD norm loss

VERSION = 'RetinaNet_DOTA2.0_GWD_NORM_1x_20220628'

"""
RetinaNet-H + gwd norm
FLOPs: 865684342;    Trainable params: 33148131

This is your evaluation result for task 1 (VOC metrics):

    mAP: 0.48849415732730095
    ap of each class: 
    plane:0.7765350866567217, 
    baseball-diamond:0.4595493895413281, 
    bridge:0.4035779839147082, 
    ground-track-field:0.5925437823720344, 
    small-vehicle:0.4091596633283188, 
    large-vehicle:0.4565196814145972, 
    ship:0.543486171740806, 
    tennis-court:0.7911472087115775, 
    basketball-court:0.6056436715449849, 
    storage-tank:0.5071997040344917, 
    soccer-ball-field:0.4514875377849248, 
    roundabout:0.5074972899976393, 
    harbor:0.449998173221673, 
    swimming-pool:0.5502158419477168, 
    helicopter:0.5276352242183413, 
    container-crane:0.1587443183942941, 
    airport:0.5122442529613705, 
    helipad:0.08970985010588972

COCO style result:

AP50: 0.48849415732730095
AP75: 0.27074929083395166
mAP: 0.27540878116565726
The submitted information is :

Description: RetinaNet_DOTA2.0_GWD_NORM_1x_20220628_104w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
