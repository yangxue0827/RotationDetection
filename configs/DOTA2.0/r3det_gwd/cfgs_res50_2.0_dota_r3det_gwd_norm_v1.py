# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
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
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0
REG_LOSS_MODE = 1

VERSION = 'RetinaNet_DOTA2.0_R3Det_GWD_NORM_2x_20220630'

"""
r3det+gwd norm
FLOPs: 1037523753;    Trainable params: 37921786

This is your evaluation result for task 1 (VOC metrics):

    mAP: 0.5005065504897706
    ap of each class: 
    plane:0.7935266075632329, 
    baseball-diamond:0.45743765692356214, 
    bridge:0.39991976928601164, 
    ground-track-field:0.6087009132278249, 
    small-vehicle:0.4341105115438158, 
    large-vehicle:0.5435969780132346, 
    ship:0.599869816100169, 
    tennis-court:0.7777466234670899, 
    basketball-court:0.5904280850553387, 
    storage-tank:0.6162996109241432, 
    soccer-ball-field:0.44326300269196517, 
    roundabout:0.4936218734409874, 
    harbor:0.44348200423726253, 
    swimming-pool:0.5503975008031091, 
    helicopter:0.430348192766536, 
    container-crane:0.13813459618766935, 
    airport:0.536459848441903, 
    helipad:0.15177431814201608

COCO style result:

AP50: 0.5005065504897706
AP75: 0.27539201854009127
mAP: 0.2821657234328435
The submitted information is :

Description: RetinaNet_DOTA2.0_R3Det_GWD_NORM_2x_20220630_104w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

