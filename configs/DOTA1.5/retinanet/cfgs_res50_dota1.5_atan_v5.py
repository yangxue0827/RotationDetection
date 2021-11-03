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
SAVE_WEIGHTS_INTE = 32000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA1.5_1x_20210813'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta)) + 180, sin^2(theta) + cos^2(theta) = 1
[-90, 90]   sin in [-1, 1]   cos in [0, 1]
FLOPs: 486656046;    Trainable params: 33099726

This is your evaluation result for task 1:

mAP: 0.5720575264888819
ap of each class: plane:0.7864624371144598, baseball-diamond:0.7756439330671105, bridge:0.393849654436808, ground-track-field:0.6186626832628273, small-vehicle:0.39361999419656213, large-vehicle:0.4092996574704831, ship:0.6423314379340461, tennis-court:0.8981020143807608, basketball-court:0.7325034724190221, storage-tank:0.6163836465326267, soccer-ball-field:0.47385185055681156, roundabout:0.6974853729014523, harbor:0.4512854606763586, swimming-pool:0.6451983907587344, helicopter:0.5270394842164826, container-crane:0.09120093389756312
The submitted information is :

Description: RetinaNet_DOTA1.5_1x_20210813_45.6w
Username: AICyber
Institute: IECAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: Yang Xue; Yang Jirui
"""



