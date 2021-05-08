# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
r3det+gwd (only refine stage) + sqrt tau=2 + data aug. + ms + res18 + 4x + 4*128conv head + mss
FLOPs: 374949903;    Trainable params: 14417960

single scale
This is your result for task 1:

mAP: 0.7104615330553556
ap of each class: plane:0.8663453594557654, baseball-diamond:0.8011578267967905, bridge:0.5197766128963681, ground-track-field:0.49668837181496467, small-vehicle:0.7573358340480076, large-vehicle:0.7753960073254165, ship:0.8610305058762486, tennis-court:0.9005286217641268, basketball-court:0.832186505829791, storage-tank:0.8230849764639923, soccer-ball-field:0.5605113168146179, roundabout:0.5885552008680969, harbor:0.6329582248058084, swimming-pool:0.6906364820204955, helicopter:0.5507311490498452
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_4x_20210122_183.6w

****************************************************************************************
multi-scale
This is your result for task 1:

mAP: 0.7354076769082152
ap of each class: plane:0.8787637070509029, baseball-diamond:0.8173156065083986, bridge:0.5176317799583069, ground-track-field:0.6920722036354392, small-vehicle:0.7378374410840408, large-vehicle:0.7778103182399576, ship:0.8645791679741641, tennis-court:0.9004563249666225, basketball-court:0.8447454366851752, storage-tank:0.8432589066889531, soccer-ball-field:0.598194089819743, roundabout:0.5973576143522173, harbor:0.6654066264571995, swimming-pool:0.6915250836100147, helicopter:0.6041608465920913
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_4x_20210122_183.6w_ms
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue

****************************************************************************************
multi-scale + mms + swa13
This is your result for task 1:

mAP: 0.7537348062114824
ap of each class: plane:0.8838267170017613, baseball-diamond:0.8475063054227168, bridge:0.5263444204777467, ground-track-field:0.7734981201313472, small-vehicle:0.7429375044195657, large-vehicle:0.7853213424747003, ship:0.8632266673606003, tennis-court:0.8911645048694999, basketball-court:0.8572513791913634, storage-tank:0.851289418716201, soccer-ball-field:0.6783590076679108, roundabout:0.5947702381829524, harbor:0.6688027698136879, swimming-pool:0.715886218565708, helicopter:0.6258374788764736
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_4x_20210122_ms_mss_swa13
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue

****************************************************************************************
mms + swa13
This is your result for task 1:

mAP: 0.7503119822569158
ap of each class: plane:0.8848939678768104, baseball-diamond:0.8341844191514508, bridge:0.5482109683577997, ground-track-field:0.7411172691924394, small-vehicle:0.7527292230905156, large-vehicle:0.7868654373012476, ship:0.8657529914418762, tennis-court:0.8927466765778316, basketball-court:0.8201604200974354, storage-tank:0.8581968930098459, soccer-ball-field:0.662238670688245, roundabout:0.6401238200112155, harbor:0.6708161527393557, swimming-pool:0.7153356025065253, helicopter:0.5813072218111404
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_4x_20210122_mss_swa13
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue


****************************************************************************************
multi-scale + mms
This is your result for task 1:

mAP: 0.7397576023156099
ap of each class: plane:0.8726924283324968, baseball-diamond:0.8259085764017077, bridge:0.5189789463543251, ground-track-field:0.7657787526388257, small-vehicle:0.7273500768230909, large-vehicle:0.7704077334265564, ship:0.8559054894650675, tennis-court:0.8917906731445102, basketball-court:0.8391248876822586, storage-tank:0.848097854944924, soccer-ball-field:0.6334066092739068, roundabout:0.5946394092851911, harbor:0.6640623304210106, swimming-pool:0.69789430382679, helicopter:0.5903259627134887
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_4x_20210122_183.6w_ms_mss
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_R3Det_GWD_4x_20210122'
NET_NAME = 'resnet18_v1b'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "1,2,3"
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
NUM_SUBNET_CONV = 4
NUM_REFINE_STAGE = 1
USE_RELU = False
FPN_CHANNEL = 128
FPN_MODE = 'fpn'

# --------------------------------------------- Anchor
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [1.]
ANCHOR_RATIOS = [1.]
ANCHOR_ANGLES = [-90, -75, -60, -45, -30, -15]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True
METHOD = 'H'
ANGLE_RANGE = 90

# -------------------------------------------- Head
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.35
IOU_NEGATIVE_THRESHOLD = 0.25
REFINE_IOU_POSITIVE_THRESHOLD = [0.5, 0.6]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.4, 0.5]

NMS = True
NMS_IOU_THRESHOLD = 0.1
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.4

# -------------------------------------------- GWD
GWD_TAU = 2.0
GWD_FUNC = tf.sqrt
