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
SAVE_WEIGHTS_INTE = 32000 * 2
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

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA1.5_RSDet_2x_20210518'

"""
RSDet-5p
FLOPs: 862193566;    Trainable params: 33051321
This is your evaluation result for task 1:

    mAP: 0.5774547328770503
    ap of each class:
    plane:0.7940904393633658,
    baseball-diamond:0.7188593189879015,
    bridge:0.3951875027384778,
    ground-track-field:0.6011415852623742,
    small-vehicle:0.4545915122459082,
    large-vehicle:0.44814418560874303,
    ship:0.6871364321703164,
    tennis-court:0.8958186065389708,
    basketball-court:0.7308610717219548,
    storage-tank:0.5946069669260221,
    soccer-ball-field:0.4655770906044697,
    roundabout:0.6924677585733299,
    harbor:0.5182629979731589,
    swimming-pool:0.6526989982691569,
    helicopter:0.48222458001340474,
    container-crane:0.10760667903525047

The submitted information is :

Description: RetinaNet_DOTA1.5_RSDet_2x_20210518_83.2w
Username: AICyber
Institute: IECAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: Yang Xue; Yang Jirui
"""


