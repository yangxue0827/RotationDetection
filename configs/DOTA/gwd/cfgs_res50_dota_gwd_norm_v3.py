# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = '0'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA'

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 3.0
REG_LOSS_MODE = 5  # GWD norm loss

VERSION = 'RetinaNet_DOTA_GWD_NORM_1x_20220625_v3'

"""
RetinaNet-H + gwd norm
FLOPs: 484913689;    Trainable params: 33002916

This is your evaluation result for task 1 (VOC metrics):

    mAP: 0.7019623409450841
    ap of each class: 
    plane:0.8889624979833605, 
    baseball-diamond:0.7772372754521742, 
    bridge:0.44875675078103616, 
    ground-track-field:0.7018370894671451, 
    small-vehicle:0.7260823005900362, 
    large-vehicle:0.644333325641051, 
    ship:0.7478534681519446, 
    tennis-court:0.9078804544761196, 
    basketball-court:0.8118186276801497, 
    storage-tank:0.8165193054416054, 
    soccer-ball-field:0.556020342820718, 
    roundabout:0.6302638094856331, 
    harbor:0.6415193368148584, 
    swimming-pool:0.6724276053150536, 
    helicopter:0.5579229240753782

COCO style result:

AP50: 0.7019623409450841
AP75: 0.407997977211734
mAP: 0.4097529935985372
The submitted information is :

Description: RetinaNet_DOTA_GWD_NORM_1x_20220625_v3_45.9w
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui
"""
