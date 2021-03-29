# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

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
USE_IOU_FACTOR = False

VERSION = 'RetinaNet_DOTA_RefineRetinaNet_2x_20201028'

"""
FLOPs: 1065231174;    Trainable params: 38691656
This is your result for task 1:

    mAP: 0.7074733381292794
    ap of each class:
    plane:0.8853694494887567,
    baseball-diamond:0.7442029035147022,
    bridge:0.4752883004198508,
    ground-track-field:0.6712068439619018,
    small-vehicle:0.7536708583696115,
    large-vehicle:0.7469206855387188,
    ship:0.8637062264616429,
    tennis-court:0.9084209821129499,
    basketball-court:0.7905644749239691,
    storage-tank:0.8187173133248197,
    soccer-ball-field:0.5824655361953132,
    roundabout:0.6013463247643736,
    harbor:0.576894321514726,
    swimming-pool:0.6766749903844729,
    helicopter:0.5166508609633821

The submitted information is :

Description: RetinaNet_DOTA_RefineRetinaNet_2x_20201028_70.2w
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

"""
