# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
r3det+gwd (only refine stage) + sqrt tau=2 + data aug. + ms + res50 + 6x + 5*256conv head + mss
FLOPs: 1368920490;    Trainable params: 40129976

single scale
This is your result for task 1:

mAP: 0.7634181600778898
ap of each class: plane:0.888207719076254, baseball-diamond:0.829435900234337, bridge:0.5562507908848845, ground-track-field:0.7275195573550398, small-vehicle:0.7851732460535438, large-vehicle:0.8310090236495815, ship:0.8745705796177181, tennis-court:0.9020813547113927, basketball-court:0.8636274681837631, storage-tank:0.8544056253270681, soccer-ball-field:0.647029596512831, roundabout:0.6141483463995911, harbor:0.7345902404435874, swimming-pool:0.7694417996682188, helicopter:0.5737811530505358
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210109_275.4w_ss
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue

******************************************************************************************
multi-scale
This is your result for task 1:

mAP: 0.7701914388454392
ap of each class: plane:0.8908994314096559, baseball-diamond:0.8412740422717381, bridge:0.5577486433080739, ground-track-field:0.7447767680141114, small-vehicle:0.777053238165341, large-vehicle:0.8299001028310027, ship:0.87568579110575, tennis-court:0.8945982453396603, basketball-court:0.8488991708356048, storage-tank:0.8567395898894529, soccer-ball-field:0.660875810286103, roundabout:0.6417324886071764, harbor:0.7513271706767073, swimming-pool:0.7535260897833023, helicopter:0.6278350001579089
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210109_275.4w_ms
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue

******************************************************************************************
multi-scale + mss
This is your result for task 1:

mAP: 0.7758189283404102
ap of each class: plane:0.8888566035867684, baseball-diamond:0.8357828324535008, bridge:0.5554398741314035, ground-track-field:0.8045909844685052, small-vehicle:0.7686217690369725, large-vehicle:0.830675754201854, ship:0.8685094810871348, tennis-court:0.8908531956357268, basketball-court:0.8308572059092085, storage-tank:0.8616945585206355, soccer-ball-field:0.7138471341943888, roundabout:0.6492505139979254, harbor:0.7620894302372899, swimming-pool:0.7323468433982865, helicopter:0.6438677442465525
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210109_275.4w_ms_mss
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue

******************************************************************************************
multi-scale + swa13
This is your result for task 1:

mAP: 0.7815217834000412
ap of each class: plane:0.8904361318323354, baseball-diamond:0.8499055814452081, bridge:0.5713677688260442, ground-track-field:0.7612774187201681, small-vehicle:0.7778998583184424, large-vehicle:0.8402890928682538, ship:0.8769943572757316, tennis-court:0.8952821602800487, basketball-court:0.8382763056232444, storage-tank:0.8564461514060031, soccer-ball-field:0.6960204454386784, roundabout:0.6374789555142868, harbor:0.76097448610277, swimming-pool:0.7921705157601568, helicopter:0.6780075215892467
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210109_275.4w_ms_swa13
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

******************************************************************************************
multi-scale + mss + swa13
This is your result for task 1:

mAP: 0.7835283621795195
ap of each class: plane:0.8843014126606585, baseball-diamond:0.8432716149764078, bridge:0.5691340085028184, ground-track-field:0.8218637425006179, small-vehicle:0.7668836737322519, large-vehicle:0.8322775965678235, ship:0.8678081079677555, tennis-court:0.8889818583950511, basketball-court:0.839317924020563, storage-tank:0.8572629680582678, soccer-ball-field:0.7207381958075649, roundabout:0.6566887627767201, harbor:0.7675711546812586, swimming-pool:0.7837317801077929, helicopter:0.6530926319372395
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210109_275.4w_ms_mss_swa13
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_R3Det_GWD_6x_20210109'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'

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

# ------------------------------------------ Train and test
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
USE_07_METRIC = True
ADD_BOX_IN_TENSORBOARD = True

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 1e-3
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(1.0 / 4.0 * SAVE_WEIGHTS_INTE)

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
NUM_SUBNET_CONV = 5
NUM_REFINE_STAGE = 1
USE_RELU = False
FPN_CHANNEL = 256
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
ANGLE_RANGE = 90

# -------------------------------------------- Head
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

NMS = True
NMS_IOU_THRESHOLD = 0.1
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.4

# -------------------------------------------- GWD
GWD_TAU = 2.0
GWD_FUNC = tf.sqrt
