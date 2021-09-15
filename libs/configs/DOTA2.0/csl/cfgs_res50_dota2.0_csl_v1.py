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

# bbox head
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 0.5
REG_LOSS_MODE = None

# CSL
LABEL_TYPE = 0  # {0: gaussian_label, 1: rectangular_label, 2: pulse_label, 3: triangle_label}
RADIUS = 6
OMEGA = 1

VERSION = 'RetinaNet_DOTA2.0_CSL_2x_20210430'

"""
gaussian label, omega=1, r=6
FLOPs: 911827508;    Trainable params: 41861031

This is your evaluation result for task 1:

mAP: 0.43319514318949137
ap of each class: plane:0.73837962181433, baseball-diamond:0.4577168668130754, bridge:0.3652826112241565, ground-track-field:0.5633578941113513, small-vehicle:0.3407092601602313, large-vehicle:0.33199600508197374, ship:0.46776685321302636, tennis-court:0.7568282160844853, basketball-court:0.5622969398961076, storage-tank:0.4952916341408789, soccer-ball-field:0.38790439733529547, roundabout:0.4748330213973504, harbor:0.38209249166809944, swimming-pool:0.5162864669224191, helicopter:0.36292846531793055, container-crane:0.07741215839375348, airport:0.39147149486663807, helipad:0.12495817896974121
The submitted information is :

Description: RetinaNet_DOTA2.0_CSL_2x_20210430_104w
Username: yangxue
Institute: UCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue
"""


