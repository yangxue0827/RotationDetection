# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 27000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA'
CLASS_NUM = 1

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

VERSION = 'FPN_Res50D_DOTA_1x_20210819'

"""
R2CNN
FLOPs: 1024153266;    Trainable params: 41772682
This is your result for task 1:

    mAP: 0.7227414456963894
    ap of each class:
    plane:0.8954291131230108,
    baseball-diamond:0.7615013248230833,
    bridge:0.47589589239010427,
    ground-track-field:0.6484503831218632,
    small-vehicle:0.7616171143029637,
    large-vehicle:0.7395101403930869,
    ship:0.8587426481796258,
    tennis-court:0.9022025499507798,
    basketball-court:0.8327346869026073,
    storage-tank:0.8431585743608815,
    soccer-ball-field:0.5106006620292729,
    roundabout:0.6561468034665185,
    harbor:0.6530002955426998,
    swimming-pool:0.6823392612570894,
    helicopter:0.6197922356022552

The submitted information is :

Description: FPN_Res50D_DOTA_1x_20201031_37.8w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
