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
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

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

VERSION = 'RetinaNet_DOTA_2x_20210401'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta))
FLOPs: 862193609;    Trainable params: 33051321
This is your result for task 1:

    mAP: 0.651696632387858
    ap of each class:
    plane:0.8879777802653781,
    baseball-diamond:0.7377507544050896,
    bridge:0.42536648303837926,
    ground-track-field:0.6532556426863868,
    small-vehicle:0.6556639819529395,
    large-vehicle:0.484385913933396,
    ship:0.7059253552779867,
    tennis-court:0.8905830467374414,
    basketball-court:0.7993635154768965,
    storage-tank:0.7555854625496603,
    soccer-ball-field:0.5261429541192497,
    roundabout:0.5980802639622864,
    harbor:0.5210216901007899,
    swimming-pool:0.6502795971050093,
    helicopter:0.4840670442069815

The submitted information is :

Description: RetinaNet_DOTA_2x_20210401_70.2w
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui
"""



