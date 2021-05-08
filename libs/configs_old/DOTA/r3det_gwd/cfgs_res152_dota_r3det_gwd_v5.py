# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
r3det+gwd (only refine stage) + sqrt tau=2 + data aug. + ms + res152 + 6x + 5*256conv head + mss
FLOPs: 4068911210;    Trainable params: 74667448

single scale
This is your result for task 1:

mAP: 0.7747016399196516
ap of each class: plane:0.8923619240527851, baseball-diamond:0.8301204757300612, bridge:0.5538368430768317, ground-track-field:0.6983532255755402, small-vehicle:0.7844455377668292, large-vehicle:0.8428541403833821, ship:0.8757512876463733, tennis-court:0.8918197015611691, basketball-court:0.8546239785166878, storage-tank:0.8615419888719404, soccer-ball-field:0.6553452719093344, roundabout:0.6338611114292596, harbor:0.746094689050884, swimming-pool:0.7865667823764704, helicopter:0.7129476408472256
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210107_226.8w_ss
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

******************************************************************************************
single scale
This is your result for task 1:

mAP: 0.7757006247637812
ap of each class: plane:0.8874079769993605, baseball-diamond:0.8263275123284586, bridge:0.5487657606775624, ground-track-field:0.7011010772141086, small-vehicle:0.788673145469387, large-vehicle:0.8459129490412377, ship:0.8737201763749018, tennis-court:0.8980701969506907, basketball-court:0.8479017664167915, storage-tank:0.8646597616329473, soccer-ball-field:0.6658365455928261, roundabout:0.6410781644818092, harbor:0.7530687875646643, swimming-pool:0.7843153631184941, helicopter:0.708670187593476
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210107_275.4w_ss
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

******************************************************************************************
multi-scale
This is your result for task 1:

mAP: 0.7843875498834784
ap of each class: plane:0.8958515355223411, baseball-diamond:0.8419285887685147, bridge:0.5653239255216435, ground-track-field:0.756880799894973, small-vehicle:0.776696484324381, large-vehicle:0.844773640450205, ship:0.8751589274090678, tennis-court:0.9005138674772389, basketball-court:0.842855439604304, storage-tank:0.8685404617089325, soccer-ball-field:0.6861049414035207, roundabout:0.6472693100385415, harbor:0.7659357916836704, swimming-pool:0.779154935804952, helicopter:0.718824598639889
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210107_275.4w_ms
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

******************************************************************************************
multi-scale + mss
This is your result for task 1:

mAP: 0.7831992115698143
ap of each class: plane:0.8899473099996086, baseball-diamond:0.8225823164232327, bridge:0.5662352871856462, ground-track-field:0.8139988477780857, small-vehicle:0.7704166536893654, large-vehicle:0.838998661648873, ship:0.8656156888989565, tennis-court:0.8897434991891202, basketball-court:0.8362894988380272, storage-tank:0.8647633097626257, soccer-ball-field:0.7044603922407522, roundabout:0.6558237570953243, harbor:0.7640663241230402, swimming-pool:0.772978314843622, helicopter:0.6920683118309329
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210107_275.4w_ms_mss
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

******************************************************************************************
multi-scale + swa13
This is your result for task 1:

mAP: 0.7891891897276755
ap of each class: plane:0.8959392331793907, baseball-diamond:0.8296014605280414, bridge:0.5882924433012168, ground-track-field:0.7503722186434625, small-vehicle:0.7762819531104056, large-vehicle:0.8483181371144024, ship:0.8730645236608151, tennis-court:0.8989253531074756, basketball-court:0.865350398406183, storage-tank:0.8681521492133617, soccer-ball-field:0.6944624820054573, roundabout:0.6593605188238154, harbor:0.7655089804105683, swimming-pool:0.7749984226932727, helicopter:0.7492095717172633
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210107_275.4w_ms_swa13
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue

******************************************************************************************
multi-scale + mss + swa13
This is your result for task 1:

mAP: 0.7907956300776425
ap of each class: plane:0.8927580303828258, baseball-diamond:0.8370275134272254, bridge:0.5926370194722532, ground-track-field:0.79852030423023, small-vehicle:0.764187458896868, large-vehicle:0.838716950360633, ship:0.865294387582269, tennis-court:0.890638404307971, basketball-court:0.8553259150262916, storage-tank:0.8650308760850987, soccer-ball-field:0.7303846298726702, roundabout:0.6755523123035938, harbor:0.7692028692787843, swimming-pool:0.7708641447635725, helicopter:0.7157936351743512
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_6x_20210107_275.4w_ms_mss_swa13
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_R3Det_GWD_6x_20210107'
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
