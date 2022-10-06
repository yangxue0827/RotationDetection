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
SAVE_WEIGHTS_INTE = 27000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CENTER_LOSS_MODE = 1  # center loss in kld
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_DOTA_KF_KL_1x_20220918_v2'

"""
RetinaNet-H + log(kl_center) + kfiou (exp(1-IoU)-1)

loss = (loss_1.reshape([n, 1]) + loss_2).reshape([n*n,1])
loss = sum(loss)
loss /= n

This is your evaluation result for task 1 (VOC metrics):

    mAP: 0.7159506032559942
    ap of each class: 
    plane:0.8956319958448199, 
    baseball-diamond:0.7853195018991056, 
    bridge:0.4565355846786333, 
    ground-track-field:0.6866889276919379, 
    small-vehicle:0.7542019498311174, 
    large-vehicle:0.7271779236220344, 
    ship:0.8447137519356132, 
    tennis-court:0.9081264105818754, 
    basketball-court:0.8137256266611681, 
    storage-tank:0.7965270049452231, 
    soccer-ball-field:0.5417951637064815, 
    roundabout:0.6304533082486643, 
    harbor:0.6442105194223064, 
    swimming-pool:0.6937147697639319, 
    helicopter:0.560436610007

COCO style result:

AP50: 0.7159506032559942
AP75: 0.39637966915928124
mAP: 0.4005802343000219
The submitted information is :

Description: RetinaNet_DOTA_KF_KL_1x_20220918_v2_35.1w
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui
"""



