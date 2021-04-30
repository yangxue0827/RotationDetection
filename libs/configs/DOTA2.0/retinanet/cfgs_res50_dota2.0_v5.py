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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = 1  # IoU-Smooth L1

VERSION = 'RetinaNet_DOTA2.0_2x_20210427'

"""
RetinaNet-H + IoU-Smooth L1
FLOPs: 676601855;    Trainable params: 33148131
This is your evaluation result for task 1:

    mAP: 0.46306689567660153
    ap of each class:
    plane:0.7754947865172579,
    baseball-diamond:0.46434700034028786,
    bridge:0.39599890161126844,
    ground-track-field:0.5841579611023893,
    small-vehicle:0.347077308109511,
    large-vehicle:0.36984795081268984,
    ship:0.47937500010188855,
    tennis-court:0.7713614052919219,
    basketball-court:0.5712051108571043,
    storage-tank:0.5361292205829321,
    soccer-ball-field:0.4011674247207251,
    roundabout:0.5077467772636136,
    harbor:0.4002952079323899,
    swimming-pool:0.5416185098631832,
    helicopter:0.5162148043881953,
    container-crane:0.13870523708490112,
    airport:0.4057909924349168,
    helipad:0.1286705231636533

The submitted information is :

Description: RetinaNet_DOTA2.0_2x_20210427_104w
Username: yangxue
Institute: UCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue
"""


