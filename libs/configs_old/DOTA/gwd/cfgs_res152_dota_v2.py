# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
RetinaNet-H + gwd fix bug + sqrt + tau=2 + data aug. + ms + res152 + 6x + 5*conv + mss
FLOPs: 3587226986;      Trainable params: 68720548

single scale
This is your result for task 1:

mAP: 0.7408985175481236
ap of each class: plane:0.8888223506945446, baseball-diamond:0.8047007035270511, bridge:0.5294462254295501, ground-track-field:0.6384858793261825, small-vehicle:0.7694794448364887, large-vehicle:0.7028213477911561, ship:0.8356015006040858, tennis-court:0.8854131132475151, basketball-court:0.8351377210733051, storage-tank:0.8493872283008324, soccer-ball-field:0.6123978957883163, roundabout:0.651303442967228, harbor:0.6545254179240159, swimming-pool:0.7169369467839489, helicopter:0.7390185449276359
The submitted information is :

Description: RetinaNet_DOTA_GWD_6x_20210107_275.4w_ss
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

******************************************************************************************
multi-scale
This is your result for task 1:

mAP: 0.7408985175481236
ap of each class: plane:0.8888223506945446, baseball-diamond:0.8047007035270511, bridge:0.5294462254295501, ground-track-field:0.6384858793261825, small-vehicle:0.7694794448364887, large-vehicle:0.7028213477911561, ship:0.8356015006040858, tennis-court:0.8854131132475151, basketball-court:0.8351377210733051, storage-tank:0.8493872283008324, soccer-ball-field:0.6123978957883163, roundabout:0.651303442967228, harbor:0.6545254179240159, swimming-pool:0.7169369467839489, helicopter:0.7390185449276359
The submitted information is :

Description: RetinaNet_DOTA_GWD_6x_20210107_275.4w_ss
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

******************************************************************************************
multi-scale + mss
This is your result for task 1:

mAP: 0.7535172301103429
ap of each class: plane:0.861372320224644, baseball-diamond:0.8158795488610238, bridge:0.5533317557978868, ground-track-field:0.7556881775339492, small-vehicle:0.7419717203969204, large-vehicle:0.6734454699174187, ship:0.8174786457427275, tennis-court:0.874838085385486, basketball-court:0.8280285328815477, storage-tank:0.8545826001529607, soccer-ball-field:0.694702581705522, roundabout:0.6720137125718214, harbor:0.7096647896904787, swimming-pool:0.7090692773121484, helicopter:0.7406912334806073
The submitted information is :

Description: RetinaNet_DOTA_GWD_6x_20210107_275.4w_ms_mss
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

******************************************************************************************
multi-scale + swa12
This is your result for task 1:

mAP: 0.7594083472779557
ap of each class: plane:0.876318295538842, baseball-diamond:0.8431922401515305, bridge:0.5482813144155365, ground-track-field:0.6998598223369994, small-vehicle:0.7617419128531924, large-vehicle:0.7012177606963704, ship:0.8312710913064713, tennis-court:0.8895723021259997, basketball-court:0.8318817729402064, storage-tank:0.8605917001981491, soccer-ball-field:0.6771926871516288, roundabout:0.6616627242773173, harbor:0.7346672206987803, swimming-pool:0.7457129941779663, helicopter:0.7279613703003465
The submitted information is :

Description: RetinaNet_DOTA_GWD_6x_20210107_275.4w_ms_swa12_0.1
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

******************************************************************************************
multi-scale + mss + swa12
This is your result for task 1:

mAP: 0.763003453536013
ap of each class: plane:0.8695796141428607, baseball-diamond:0.8388495065280341, bridge:0.543561334164674, ground-track-field:0.7753060397299352, small-vehicle:0.7440574316542253, large-vehicle:0.6848076677716322, ship:0.8033843592349189, tennis-court:0.8661680594650631, basketball-court:0.8340668176456562, storage-tank:0.8555236997871242, soccer-ball-field:0.7347255464759853, roundabout:0.6777458706465952, harbor:0.7256648720336767, swimming-pool:0.7575792736307899, helicopter:0.7340317101290215
The submitted information is :

Description: RetinaNet_DOTA_GWD_6x_20210107_275.4w_ms_mss_swa12_0.1
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_GWD_6x_20210107'
NET_NAME = 'resnet152_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2,3,4,5,6,7"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 27000 * 6

SUMMARY_PATH = os.path.join(ROOT_PATH, 'output/summary')
TEST_SAVE_PATH = os.path.join(ROOT_PATH, 'tools/test_result')

pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_R_DIR = os.path.join(ROOT_PATH, 'output/evaluate_result_pickle/')

# ------------------------------------------ Train and Test
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
USE_07_METRIC = True
ADD_BOX_IN_TENSORBOARD = True

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0
REG_LOSS_MODE = 2
ALPHA = 1.0
BETA = 1.0

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 1e-3
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(1.0 / 8.0 * SAVE_WEIGHTS_INTE)

# -------------------------------------------- Dataset
DATASET_NAME = 'DOTA'  # 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = [800, 450, 500, 640, 700, 900, 1000, 1100, 1200]
IMG_MAX_LENGTH = 1200
CLASS_NUM = 15

IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = True

# --------------------------------------------- Network
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 1e-4
USE_GN = False
FPN_CHANNEL = 256
NUM_SUBNET_CONV = 5
FPN_MODE = 'fpn'

# --------------------------------------------- Anchor
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [1, 1 / 2, 2., 1 / 3., 3., 5., 1 / 5.]
ANCHOR_ANGLES = [-90, -75, -60, -45, -30, -15]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True
METHOD = 'H'
USE_ANGLE_COND = False
ANGLE_RANGE = 90  # or 180

# -------------------------------------------- Head
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4

NMS = True
NMS_IOU_THRESHOLD = 0.1
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.4

# -------------------------------------------- GWD
GWD_TAU = 2.0
GWD_FUNC = tf.sqrt
