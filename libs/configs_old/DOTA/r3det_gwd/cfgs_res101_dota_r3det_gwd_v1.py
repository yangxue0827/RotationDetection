# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
r3det+gwd (only refine stage) + sqrt tau=2 + data aug. + ms + res101 + 3x
FLOPs: 1486599773;    Trainable params: 56709560

single scale
This is your result for task 1:

mAP: 0.7507929219725538
ap of each class:
plane:0.8959363254654953,
baseball-diamond:0.8117970905483454,
bridge:0.528941868308621,
ground-track-field:0.7036828276226571,
small-vehicle:0.7773608512825836,
large-vehicle:0.824193538909995,
ship:0.869907508286484,
tennis-court:0.8931429748331345,
basketball-court:0.8306405809724954,
storage-tank:0.8596700800149586,
soccer-ball-field:0.6406646930436465,
roundabout:0.6513733145761638,
harbor:0.6805467355307225,
swimming-pool:0.7095327160387952,
helicopter:0.5845027241542098
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201223_137.7w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

multi-scale
This is your result for task 1:

mAP: 0.7565987110400665
ap of each class:
plane:0.8964434495830199,
baseball-diamond:0.8170201102441278,
bridge:0.5252222485821921,
ground-track-field:0.729602202372266,
small-vehicle:0.7601882164495246,
large-vehicle:0.8260373681497573,
ship:0.8716819767827122,
tennis-court:0.895671606749032,
basketball-court:0.8124548266228695,
storage-tank:0.8608847379716957,
soccer-ball-field:0.6224300564022891,
roundabout:0.657426649146108,
harbor:0.6805129469115909,
swimming-pool:0.7495597236660946,
helicopter:0.6438445459677189

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201223_137.7w_ms
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

multi-scale + flip
This is your result for task 1:

mAP: 0.7579683916708654
ap of each class:
plane:0.8952961695089519,
baseball-diamond:0.8194477842585369,
bridge:0.5243575576743463,
ground-track-field:0.7159062303762178,
small-vehicle:0.7522987121676139,
large-vehicle:0.8282767251249042,
ship:0.8719194994284161,
tennis-court:0.8900495351735876,
basketball-court:0.8399873550181818,
storage-tank:0.8593060497981101,
soccer-ball-field:0.6213106308173056,
roundabout:0.6531238042666215,
harbor:0.7089754248166696,
swimming-pool:0.7442537008416809,
helicopter:0.6450166957918368

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201223_137.7w_ms_f
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

multi-scale + mss
This is your result for task 1:

mAP: 0.7622308686293031
ap of each class: plane:0.8956285239247809, baseball-diamond:0.8123129086084642, bridge:0.5338483720916204, ground-track-field:0.7937988239995809, small-vehicle:0.7511751772047529, large-vehicle:0.8213795039925708, ship:0.8685701145257962, tennis-court:0.888685607876141, basketball-court:0.8121412656739693, storage-tank:0.8628395323661605, soccer-ball-field:0.6536075407437665, roundabout:0.650555639430081, harbor:0.728762235286426, swimming-pool:0.7304094256459305, helicopter:0.6297483580695049
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201223_137.7w_ms_mss
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

------------------------------------------------------------------------
SWA6 + single scale
This is your result for task 1:

mAP: 0.7536616140168031
ap of each class:
plane:0.8952691912932887,
baseball-diamond:0.820224997940994,
bridge:0.5351902682975013,
ground-track-field:0.6990760862785812,
small-vehicle:0.7768718792058928,
large-vehicle:0.8311637947866269,
ship:0.8707906667992975,
tennis-court:0.9041214885985671,
basketball-court:0.844673715234245,
storage-tank:0.8622850403351549,
soccer-ball-field:0.652075771320803,
roundabout:0.6344222159586387,
harbor:0.6796953794936931,
swimming-pool:0.7077122802895897,
helicopter:0.5913514344191727

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201223_137.7w_s_swa6
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

SWA9 + single scale
mAP: 0.7541074969944405
ap of each class:
plane:0.8945929006004023,
baseball-diamond:0.8239698637790639,
bridge:0.5375054158208356,
ground-track-field:0.707263661407391,
small-vehicle:0.7784868305276835,
large-vehicle:0.8313093170826968,
ship:0.8716851894984969,
tennis-court:0.894339902291634,
basketball-court:0.8124139096196356,
storage-tank:0.8621962310027806,
soccer-ball-field:0.6560753654400574,
roundabout:0.6425826383892252,
harbor:0.6781700792445191,
swimming-pool:0.7540293197758513,
helicopter:0.5669918304363327

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201223_137.7w_s_swa9
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue


SWA9 + multi-scale
This is your result for task 1:

mAP: 0.7611382693726101
ap of each class:
plane:0.8965990887138806,
baseball-diamond:0.8211375961066281,
bridge:0.5274322763152262,
ground-track-field:0.7164191574732031,
small-vehicle:0.7594817184692972,
large-vehicle:0.8309449959450008,
ship:0.8697248627752349,
ennis-court:0.8927766089227556,
basketball-court:0.8504259904102585,
storage-tank:0.8616896266258318,
soccer-ball-field:0.6551601152374349,
roundabout:0.6329228740725862,
harbor:0.7218286216219233,
swimming-pool:0.748803742710907,
helicopter:0.6317267651889841

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201223_137.7w_ms_swa9
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue


SWA9 + multi-scale + mss
This is your result for task 1:

mAP: 0.7667344567887646
ap of each class: plane:0.8932663066411421, baseball-diamond:0.8086480184788661, bridge:0.5328287201353856, ground-track-field:0.782914642745032, small-vehicle:0.7539546562133567, large-vehicle:0.8268751115449491, ship:0.8708668707425938, tennis-court:0.8934725656178293, basketball-court:0.826406161849139, storage-tank:0.8640553822619391, soccer-ball-field:0.6984796440184639, roundabout:0.6471336736334592, harbor:0.7418605732124637, swimming-pool:0.7617818475817874, helicopter:0.5984726771550614
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201223_137.7w_ms_swa9_mss
Username: liuqingiqng
Institute: Central South University
Emailadress: liuqingqing@csu.edu.cn
TeamMembers: liuqingqing


"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_R3Det_GWD_3x_20201223'
NET_NAME = 'resnet101_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 27000 * 3

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
IMG_SHORT_SIDE_LEN = [800, 640, 700, 900, 1000, 1100]
IMG_MAX_LENGTH = 1100
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
NUM_SUBNET_CONV = 4
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
