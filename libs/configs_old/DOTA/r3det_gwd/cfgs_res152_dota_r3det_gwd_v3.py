# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
r3det+gwd (only refine stage) + sqrt tau=2 + data aug. + ms + res152 + 6x + 5*256conv head

multi-scale+flip
This is your result for task 1:

mAP: 0.7787093225719842
ap of each class:
plane:0.89461770093155,
baseball-diamond:0.8236018667686434,
bridge:0.5021414308765855,
ground-track-field:0.7435082801347662,
small-vehicle:0.7857255649492857,
large-vehicle:0.8382285774947096,
ship:0.8757862302153405,
tennis-court:0.8987684849725746,
basketball-court:0.8508137596719582,
storage-tank:0.868271286694195,
soccer-ball-field:0.6838162701460044,
roundabout:0.6370974555832092,
harbor:0.7542997372314082,
swimming-pool:0.7673376041149634,
helicopter:0.7566255887945696

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210101_291.6w_ms_f
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

multi-scale
This is your result for task 1:

mAP: 0.781053817845219
ap of each class:
plane:0.8916637153713546,
baseball-diamond:0.8404992598575666,
bridge:0.514273743753299,
ground-track-field:0.7440553692445901,
small-vehicle:0.7861094590168505,
large-vehicle:0.8431559832540744,
ship:0.8751741786815641,
tennis-court:0.896638838767339,
basketball-court:0.8516147950056526,
storage-tank:0.8679971722386121,
soccer-ball-field:0.6862493149004335,
roundabout:0.667533488588698,
harbor:0.7666314410601642,
swimming-pool:0.7725622344190313,
helicopter:0.7116482735190545

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210101_356.4w_ms
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue

multi-scale + swa5
This is your result for task 1:

mAP: 0.7846967729427209
ap of each class: plane:0.8919752020840401, baseball-diamond:0.832536141609128, bridge:0.5465600172547991, ground-track-field:0.7542388549761907, small-vehicle:0.7872751842388821, large-vehicle:0.8453935228446343, ship:0.8750340527454509, tennis-court:0.8908861239826896, basketball-court:0.8525540826454344, storage-tank:0.8675278695981044, soccer-ball-field:0.6893308041623426, roundabout:0.6497655415208251, harbor:0.7654674663658851, swimming-pool:0.7825265696577711, helicopter:0.7393801604546356
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210101_356.4w_ms_swa10
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

multi-scale + swa10
This is your result for task 1:

mAP: 0.7851001761526747
ap of each class: plane:0.8926517173170925, baseball-diamond:0.8382546327120208, bridge:0.5351538209043163, ground-track-field:0.7587322787466556, small-vehicle:0.78632256233217, large-vehicle:0.843705577938495, ship:0.8739304239920611, tennis-court:0.8974093700017866, basketball-court:0.8445686866573267, storage-tank:0.8657673140300304, soccer-ball-field:0.6765772398377256, roundabout:0.6677921239841383, harbor:0.7619364299879183, swimming-pool:0.7871213290604916, helicopter:0.7465791347878928
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210101_356.4w_ms_swa5
Username: liuqingiqng
Institute: Central South University
Emailadress: liuqingqing@csu.edu.cn
TeamMembers: liuqingqing


multi-scale + swa10 + mss
This is your result for task 1:

mAP: 0.7876536388287331
ap of each class:
plane:0.8892676182348125,
baseball-diamond:0.8380399764265094,
bridge:0.5417931344048306,
ground-track-field:0.8047689607468232,
small-vehicle:0.7742709708367945,
large-vehicle:0.8384061252976336,
ship:0.8667803304802113,
tennis-court:0.8825490700386722,
basketball-court:0.849107009356962,
storage-tank:0.8635221779812626,
soccer-ball-field:0.718231766749811,
roundabout:0.671254360392042,
harbor:0.7670048672500374,
swimming-pool:0.7766585596416251,
helicopter:0.7331496545929699
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210101_356.4w_ms_swa10_mss
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue


single scale
This is your result for task 1:

mAP: 0.7726868436788519
ap of each class:
plane:0.890554603183652,
baseball-diamond:0.8328498635978271,
bridge:0.5275470846341096,
ground-track-field:0.6972616812234654,
small-vehicle:0.796455271567893,
large-vehicle:0.8374070475478563,
ship:0.8791685072741147,
tennis-court:0.891880103555982,
basketball-court:0.839987919843173,
storage-tank:0.8674737684273185,
soccer-ball-field:0.6625363589195475,
roundabout:0.6819144040462316,
harbor:0.7588559440508911,
swimming-pool:0.7642584786452707,
helicopter:0.6621516186654459

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210101_356.4w_ss
Username: liuqingiqng
Institute: Central South University
Emailadress: liuqingqing@csu.edu.cn
TeamMembers: liuqingqing


single scale 800-200
This is your result for task 1:

mAP: 0.7756815407317015
ap of each class:
plane:0.8884095794629929,
baseball-diamond:0.8370358080077156,
bridge:0.5295324501460803,
ground-track-field:0.6946597867417675,
small-vehicle:0.7962974069553106,
large-vehicle:0.8384071004202205,
ship:0.8783191492251149,
tennis-court:0.8973098987336423,
basketball-court:0.8499055021679619,
storage-tank:0.8671241392841263,
soccer-ball-field:0.6872967384407853,
roundabout:0.682377503221729,
harbor:0.7577828422591375,
swimming-pool:0.761615541499773,
helicopter:0.6691496644091638

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210101_356.4w_s_800_200
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue

"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_R3Det_GWD_6x_20210101'
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
