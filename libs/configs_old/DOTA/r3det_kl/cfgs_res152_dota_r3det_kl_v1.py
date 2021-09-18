# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
r3det + kl + sqrt tau=2 + res152 + ms + mss + data aug. + 4x + swa + 4*256conv head
FLOPs: 1860941849;    Trainable params: 72307128
------------------------------------------------------------------------------------

This is your result for task 1:

mAP: 0.7644890814759712
ap of each class: plane:0.8893045841194216, baseball-diamond:0.8208948792824698, bridge:0.5365300871879102, ground-track-field:0.6850496423317299, small-vehicle:0.7773029802959452, large-vehicle:0.8427024259479862, ship:0.874217254357821, tennis-court:0.8936204288787335, basketball-court:0.8227906013065533, storage-tank:0.8588636268891325, soccer-ball-field:0.6567932311086844, roundabout:0.6443595655944814, harbor:0.7555527599301144, swimming-pool:0.7687322366509484, helicopter:0.6406219182576353
The submitted information is :

Description: RetinaNet_DOTA_R3Det_KL_4x_20210212_183.6w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
------------------------------------------------------------------------------------
ms
This is your result for task 1:

mAP: 0.7720934062378758
ap of each class: plane:0.86956004032608, baseball-diamond:0.8360984821154138, bridge:0.5488602224197876, ground-track-field:0.7438299164928236, small-vehicle:0.7712808234704196, large-vehicle:0.8437502431298498, ship:0.8733260607112842, tennis-court:0.8966454438442621, basketball-court:0.8628597660874431, storage-tank:0.8665167313363151, soccer-ball-field:0.6708465294072964, roundabout:0.6228304169462704, harbor:0.7643356358767779, swimming-pool:0.7545345956154696, helicopter:0.6561261857886455
The submitted information is :

Description: RetinaNet_DOTA_R3Det_KL_4x_20210212_183.6w_ms
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
------------------------------------------------------------------------------------
mss
This is your result for task 1:

mAP: 0.777343221756977
ap of each class: plane:0.887259168967991, baseball-diamond:0.8252935229983368, bridge:0.5461250341741551, ground-track-field:0.7942264278168191, small-vehicle:0.7775117500788149, large-vehicle:0.8323063286171692, ship:0.8695977703579528, tennis-court:0.8913164814976192, basketball-court:0.8399040976907813, storage-tank:0.8645906597843693, soccer-ball-field:0.6656812289328394, roundabout:0.6625880824277949, harbor:0.762318042598768, swimming-pool:0.7881855035229847, helicopter:0.653244226888257
The submitted information is :

Description: RetinaNet_DOTA_R3Det_KL_4x_20210212_172.8w_mss
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

------------------------------------------------------------------------------------

This is your result for task 1:

mAP: 0.771531200968382
ap of each class: plane:0.8630730123408624, baseball-diamond:0.8311408534298542, bridge:0.5473581978235935, ground-track-field:0.7299158178578212, small-vehicle:0.769740822741742, large-vehicle:0.8427384822602759, ship:0.8728610853988574, tennis-court:0.8924242975040919, basketball-court:0.873278506705244, storage-tank:0.8699508011611026, soccer-ball-field:0.6754658207812873, roundabout:0.6173511967585184, harbor:0.7645432359934285, swimming-pool:0.7668395898337872, helicopter:0.6562862939352613
The submitted information is :

Description: RetinaNet_DOTA_R3Det_KL_4x_20210212_183.6w_ms_mss_14swa
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_R3Det_KL_4x_20210212'
NET_NAME = 'resnet152_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 27000 * 4

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
IMG_SHORT_SIDE_LEN = [800, 450, 500, 640, 700, 900, 1000]
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

# -------------------------------------------- KLD
KL_TAU = 2.0
KL_FUNC = 0
