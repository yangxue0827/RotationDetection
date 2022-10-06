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
SAVE_WEIGHTS_INTE = 40000
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
CENTER_LOSS_MODE = 1  # center loss in kld
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_DOTA2.0_KF_1x_20220920'

"""
RetinaNet-H + log(kl_center) + kfiou (exp(1-IoU)-1)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

FLOPs: 487527642;    Trainable params: 33148131

This is your evaluation result for task 1 (VOC metrics):

mAP: 0.4894285127967532
ap of each class: 

    plane:0.7855100097732065, 
    baseball-diamond:0.4679762971260954, 
    bridge:0.39910657454410275, 
    ground-track-field:0.5978173014224926, 
    small-vehicle:0.418202632274417, 
    large-vehicle:0.4827424581806402, 
    ship:0.5763476713309732, 
    tennis-court:0.7866893867367266, 
    basketball-court:0.5900664388926606, 
    storage-tank:0.522142612448238, 
    soccer-ball-field:0.4212896657422925, 
    roundabout:0.5226779625553494, 
    harbor:0.4416480396458489, 
    swimming-pool:0.5307778020790656, 
    helicopter:0.5099261922359674, 
    container-crane:0.12000347981943073, 
    airport:0.5076951313967711, 
    helipad:0.1290935741372789

COCO style result:

AP50: 0.4894285127967532
AP75: 0.24352812425730352
mAP: 0.26443484834115494
The submitted information is :

Description: RetinaNet_DOTA2.0_KF_1x_20220920_52w
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



