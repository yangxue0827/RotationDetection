# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
RetinaNet-H + kl + sqrt + tau=2 + data aug. + ms + res152 + 6x + 5*conv + mss
FLOPs: 1731833968;    Trainable params: 68720548
------------------------------------------------------------------------------------------------------------------
This is your result for task 1:

mAP: 0.7557308230724562
ap of each class: plane:0.8887785597969426, baseball-diamond:0.829515957482372, bridge:0.5053613045899416, ground-track-field:0.6830540905062377, small-vehicle:0.7815565920778175, large-vehicle:0.759221121355599, ship:0.8464599890476963, tennis-court:0.8943438735516858, basketball-court:0.8555829104648072, storage-tank:0.8442670279916429, soccer-ball-field:0.6476566627161651, roundabout:0.6307209644817737, harbor:0.7597113857359125, swimming-pool:0.7440092131243214, helicopter:0.6657226931639265
The submitted information is :

Description: RetinaNet_DOTA_KL_6x_20210221_275.4w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

------------------------------------------------------------------------------------------------------------------
ms
This is your result for task 1:

mAP: 0.7656018228247946
ap of each class: plane:0.8836539862503108, baseball-diamond:0.847444151859026, bridge:0.5127399695118523, ground-track-field:0.7229617428791448, small-vehicle:0.7724845578295932, large-vehicle:0.7658016876661938, ship:0.8388764813767275, tennis-court:0.8950853104202099, basketball-court:0.8516988649932271, storage-tank:0.8546280944070542, soccer-ball-field:0.6892725967403556, roundabout:0.6402004524564894, harbor:0.7610972069021169, swimming-pool:0.7638232409119834, helicopter:0.6842589981676335
The submitted information is :

Description: RetinaNet_DOTA_KL_6x_20210221_275.4w_ms
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui
------------------------------------------------------------------------------------------------------------------
ms+swa12
This is your result for task 1:

mAP: 0.7676542040747402
ap of each class: plane:0.8849768296604477, baseball-diamond:0.8366293229659247, bridge:0.5091808790642554, ground-track-field:0.723991326883, small-vehicle:0.7775678941812224, large-vehicle:0.7757413897343344, ship:0.839245418765504, tennis-court:0.8878681212514024, basketball-court:0.8551885618774783, storage-tank:0.8587844225116738, soccer-ball-field:0.6880326947902784, roundabout:0.615772553644298, harbor:0.7571701105191584, swimming-pool:0.7804533634109798, helicopter:0.7242101718611471
The submitted information is :

Description: RetinaNet_DOTA_KL_6x_20210221_275.4w_ms_swa12
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui
------------------------------------------------------------------------------------------------------------------
ms + mss
This is your result for task 1:

mAP: 0.7646368387981715
ap of each class: plane:0.8737428874619758, baseball-diamond:0.8455768361521168, bridge:0.5076160939634508, ground-track-field:0.7854019450822015, small-vehicle:0.7566150824569645, large-vehicle:0.7466335113701572, ship:0.8188491496838732, tennis-court:0.8810293827403389, basketball-court:0.8366894304352066, storage-tank:0.8510784426222289, soccer-ball-field:0.713450095809895, roundabout:0.6584115434617008, harbor:0.7427809305238757, swimming-pool:0.7679717968322852, helicopter:0.6837054533762996
The submitted information is :

Description: RetinaNet_DOTA_KL_6x_20210221_275.4w_ms_mss
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

------------------------------------------------------------------------------------------------------------------
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_KL_6x_20210221'
NET_NAME = 'resnet152_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2"
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
REG_LOSS_MODE = 3
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
IMG_SHORT_SIDE_LEN = [800, 450, 500, 640, 700, 900, 1000]
IMG_MAX_LENGTH = 1000
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

# -------------------------------------------- KLD
KL_TAU = 2.0
KL_FUNC = 0
