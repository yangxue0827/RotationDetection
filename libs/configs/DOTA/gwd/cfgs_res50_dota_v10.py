# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = '0'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0
REG_LOSS_MODE = 2  # GWD loss

GWD_TAU = 2.0
GWD_FUNC = tf.sqrt

VERSION = 'RetinaNet_DOTA_1x_20201217'

"""
RetinaNet-H + gwd fix bug + sqrt + tau=2
FLOPs: 484911755;    Trainable params: 33002916
This is your result for task 1:

mAP: 0.6893230234209595
ap of each class:
plane:0.884893463362855,
baseball-diamond:0.7787777797702516,
bridge:0.44074539967179616,
ground-track-field:0.6608440803892738,
small-vehicle:0.7191979620799155,
large-vehicle:0.6255851085159841,
ship:0.779426808679221,
tennis-court:0.8974576395890794,
basketball-court:0.814347193904471,
storage-tank:0.7964462198249043,
soccer-ball-field:0.5230179777462076,
roundabout:0.6351989821343238,
harbor:0.6024971142923126,
swimming-pool:0.6651492134481283,
helicopter:0.5162604079056684

The submitted information is :

Description: RetinaNet_DOTA_1x_20201217_45.9w
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

SWA
This is your result for task 1:

mAP: 0.6991693603891261
ap of each class: plane:0.88600993345841,
baseball-diamond:0.7858734185680321,
bridge:0.4410165471752093,
ground-track-field:0.6723616914240896,
small-vehicle:0.7077068065357551,
large-vehicle:0.6254285401001521,
ship:0.7977532472401473,
tennis-court:0.8885662573484675,
basketball-court:0.8191731301876931,
storage-tank:0.8045513092617368,
soccer-ball-field:0.5744355017478517,
roundabout:0.6401702403997233,
harbor:0.6264180463456216,
swimming-pool:0.6651656639513736,
helicopter:0.5529100720926275
The submitted information is :

Description: RetinaNet_DOTA_1x_20201217_45.9w_swa_12
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""
