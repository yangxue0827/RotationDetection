# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_EPOCH = [36, 48, 60]
MAX_EPOCH = 51
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
IMG_SHORT_SIDE_LEN = [800, 450, 500, 640, 700, 900, 1000, 1100, 1200]
IMG_MAX_LENGTH = 1200
# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = True

# model
NET_NAME = 'resnet152_v1d'
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_SUBNET_CONV = 5

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_DOTA_KF_6x_20210907'

"""
RetinaNet-H + KF
FLOPs: 1731839729;    Trainable params: 68720548

This is your result for task 1:

mAP: 0.755632394336779
ap of each class: plane:0.8933402270054778, baseball-diamond:0.850326114261476, bridge:0.5290861458081795, ground-track-field:0.7092333283364275, small-vehicle:0.7721899813973914, large-vehicle:0.6999888360106596, ship:0.8222061431610613, tennis-court:0.908440853005922, basketball-court:0.8774387571631029, storage-tank:0.8477200436877893, soccer-ball-field:0.6288263884286562, roundabout:0.6338656572930002, harbor:0.7506783819725362, swimming-pool:0.7097503892211662, helicopter:0.701394668298837
The submitted information is :

Description: RetinaNet_DOTA_KF_6x_20210907_swa12
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui


This is your result for task 1:

    mAP: 0.7714480657250247
    ap of each class:
    plane:0.8924651350114676,
    baseball-diamond:0.8539214600590666,
    bridge:0.5469541915788453,
    ground-track-field:0.7956635840492524,
    small-vehicle:0.7683475238165155,
    large-vehicle:0.6940226676410128,
    ship:0.8198128092282506,
    tennis-court:0.9083348812929937,
    basketball-court:0.8743851752286917,
    storage-tank:0.8581258183799104,
    soccer-ball-field:0.6998806456889523,
    roundabout:0.6782773594363836,
    harbor:0.7518643028421848,
    swimming-pool:0.7302444015971608,
    helicopter:0.6994210300246804

The submitted information is :

Description: RetinaNet_DOTA_KF_6x_20210907_264.6w_mss
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

This is your result for task 1:

    mAP: 0.7735061371652804
    ap of each class:
    plane:0.8946292467365491,
    baseball-diamond:0.8571538990849707,
    bridge:0.5493885008760021,
    ground-track-field:0.8037312729466313,
    small-vehicle:0.7716442419204482,
    large-vehicle:0.6923028876282339,
    ship:0.8090237474469362,
    tennis-court:0.9078589847071914,
    basketball-court:0.8778531066822405,
    storage-tank:0.8612918968231646,
    soccer-ball-field:0.73632349033733613,
    roundabout:0.6811073386835343,
    harbor:0.7523342863122933,
    swimming-pool:0.7160951131350827,
    helicopter:0.6949426311225669

The submitted information is :

Description: RetinaNet_DOTA_KF_6x_20210907_swa12_mss
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

"""
