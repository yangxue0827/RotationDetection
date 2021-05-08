# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
r3det+gwd (only refine stage) + sqrt tau=2 + data aug. + ms + res152 + 3x

multi-scale + mss
This is your result for task 1:

mAP: 0.7694868920869342
ap of each class: plane:0.8961593861476674, baseball-diamond:0.8227137445423284, bridge:0.5234537760655572, ground-track-field:0.7730328497323228, small-vehicle:0.7694531856952423, large-vehicle:0.8252925230988806, ship:0.8720111474913582, tennis-court:0.890786942183883, basketball-court:0.8457831463630802, storage-tank:0.8620875692024108, soccer-ball-field:0.6521020353061363, roundabout:0.6446156994628482, harbor:0.7498720313436928, swimming-pool:0.7630177430634582, helicopter:0.651921601605146
The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201225_162w_ms_mss
Username: DetectionTeamCSU
Institute: CSU
Emailadress: yangxue@csu.edu.cn
TeamMembers: YangXue

multi-scale
This is your result for task 1:

mAP: 0.7618432817590247
ap of each class:
plane:0.8954797060874107,
baseball-diamond:0.8228498004740165,
bridge:0.5238744049918231,
ground-track-field:0.6830342722389584,
small-vehicle:0.7785510775369009,
large-vehicle:0.8340248411970193,
ship:0.8748448441443502,
tennis-court:0.8955878169404766,
basketball-court:0.8426996732159534,
storage-tank:0.861355161286582,
soccer-ball-field:0.6538458079639438,
roundabout:0.6324989994483515,
harbor:0.7133023727152026,
swimming-pool:0.7235538022206461,
helicopter:0.6921466459237358

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201225_162w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

single scale
This is your result for task 1:

mAP: 0.7634904667286719
ap of each class:
plane:0.8950765575286099,
baseball-diamond:0.8267722383803965,
bridge:0.5192437194082951,
ground-track-field:0.6951273537505667,
small-vehicle:0.7896574989053226,
large-vehicle:0.8337646097239736,
ship:0.8752598829154427,
tennis-court:0.8967230029180067,
basketball-court:0.8564663259750458,
storage-tank:0.8616994567401048,
soccer-ball-field:0.6390397096934775,
roundabout:0.6743588172131516,
harbor:0.682652783673797,
swimming-pool:0.7643156405415441,
helicopter:0.6421994035623451

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201225_162w_ss
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui


single scale 800-200
This is your result for task 1:

mAP: 0.7468850767650241
ap of each class:
plane:0.8942071101712381,
baseball-diamond:0.8206546825901471,
bridge:0.5243710460681006,
ground-track-field:0.7007756076081786,
small-vehicle:0.7710977805718382,
large-vehicle:0.7660771273105712,
ship:0.8631883715159443,
tennis-court:0.897925909495094,
basketball-court:0.8212988079250682,
storage-tank:0.8610130530594753,
soccer-ball-field:0.6270688037492362,
roundabout:0.6611295907270678,
harbor:0.6777232068205228,
swimming-pool:0.7167061095166871,
helicopter:0.6000389443461915

The submitted information is :

Description: RetinaNet_DOTA_R3Det_GWD_3x_20201225_162w_ss_800_200
Username: yangxue
Institute: DetectionTeamUCAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: yangxue, yangjirui

"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_R3Det_GWD_3x_20201225'
NET_NAME = 'resnet152_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2,3,4,5,6,7"
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
IMG_SHORT_SIDE_LEN = [800, 640, 700, 900, 1000, 1100, 1200]
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
