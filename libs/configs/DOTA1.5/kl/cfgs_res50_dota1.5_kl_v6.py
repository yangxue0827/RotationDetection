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
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0
REG_LOSS_MODE = 3  # KLD loss

KL_TAU = 1.0
KL_FUNC = 1   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA1.5_KL_2x_20210318'

"""
RetinaNet-H + kl + log + tau=1
FLOPs: 862193662;    Trainable params: 33051321

This is your evaluation result for task 1:

mAP: 0.6249647279068532
ap of each class: 
plane:0.7946632258381133, 
baseball-diamond:0.7649378291166915, 
bridge:0.41981691015478395, 
ground-track-field:0.6527771749645889, 
small-vehicle:0.504286646635626, 
large-vehicle:0.6991134179610486, 
ship:0.8209201265856977, 
tennis-court:0.9034872805869663, 
basketball-court:0.7467651089670718, 
storage-tank:0.5889356328175649, 
soccer-ball-field:0.515672563776736, 
roundabout:0.6611450477287045, 
harbor:0.6407224821333771, 
swimming-pool:0.640407822337024, 
helicopter:0.5333441855180986, 
container-crane:0.11244019138755981

The submitted information is :

Description: RetinaNet_DOTA1.5_KL_2x_20210318_83.2w
Username: AICyber
Institute: IECAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: Yang Xue; Yang Jirui
"""

