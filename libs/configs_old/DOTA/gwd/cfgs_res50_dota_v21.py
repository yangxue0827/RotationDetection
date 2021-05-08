# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
RetinaNet-H + gwd fix bug + sqrt + tau=2 + train set
FLOPs: 1048220867;    Trainable params: 33002916
{'0.6': {'baseball-diamond': 0.6144571737940573, 'mAP': 0.6041576188991775, 'basketball-court': 0.6273976478518356, 'ship': 0.7466697449513071, 'roundabout': 0.6074065946882581, 'plane': 0.8921289310031991, 'harbor': 0.419929273618182, 'large-vehicle': 0.5594159121370192, 'soccer-ball-field': 0.6264710031524907, 'swimming-pool': 0.4619379286447885, 'helicopter': 0.44330776785866605, 'small-vehicle': 0.5650309829882413, 'tennis-court': 0.9003347133310372, 'ground-track-field': 0.5330025855269901, 'bridge': 0.29104019837228257, 'storage-tank': 0.7738338255693089}, '0.5': {'baseball-diamond': 0.6782710563159899, 'mAP': 0.6544813532872892, 'basketball-court': 0.6413466618008495, 'ship': 0.8268875675010192, 'roundabout': 0.659982065341791, 'plane': 0.8953406352844886, 'harbor': 0.5865021413981998, 'large-vehicle': 0.6458481363921764, 'soccer-ball-field': 0.6612376121697052, 'swimming-pool': 0.5441352970612305, 'helicopter': 0.4706360154281975, 'small-vehicle': 0.597336519078226, 'tennis-court': 0.9010162451888205, 'ground-track-field': 0.5529821034445813, 'bridge': 0.37269324116722713, 'storage-tank': 0.7830050017368321}, '0.65': {'baseball-diamond': 0.5657919023960288, 'mAP': 0.5695428973362071, 'basketball-court': 0.5941862722011403, 'ship': 0.7347515994291861, 'roundabout': 0.5884684177604006, 'plane': 0.8870436837828752, 'harbor': 0.32679140914698496, 'large-vehicle': 0.5292245692361258, 'soccer-ball-field': 0.5803693673587923, 'swimming-pool': 0.36534006298252336, 'helicopter': 0.43988603988603986, 'small-vehicle': 0.5169756154624324, 'tennis-court': 0.9003347133310372, 'ground-track-field': 0.5022331289347248, 'bridge': 0.2483574646746264, 'storage-tank': 0.763389213460187}, '0.75': {'baseball-diamond': 0.21714647649773017, 'mAP': 0.40733760250250906, 'basketball-court': 0.5538482995710056, 'ship': 0.5820298598488817, 'roundabout': 0.40053361832349976, 'plane': 0.7588270608289814, 'harbor': 0.15883970260839905, 'large-vehicle': 0.34271625247452486, 'soccer-ball-field': 0.3731964587720783, 'swimming-pool': 0.12801079668360388, 'helicopter': 0.2636630912946703, 'small-vehicle': 0.3210452277205892, 'tennis-court': 0.8946596093809345, 'ground-track-field': 0.3845534783795932, 'bridge': 0.09123465683965531, 'storage-tank': 0.6397594483134862}, '0.8': {'baseball-diamond': 0.10876033057851239, 'mAP': 0.3055668248399325, 'basketball-court': 0.48656905496340075, 'ship': 0.38194228495426624, 'roundabout': 0.24756142959727737, 'plane': 0.6168270508659637, 'harbor': 0.10888449299045326, 'large-vehicle': 0.22877932122044264, 'soccer-ball-field': 0.2918547271488448, 'swimming-pool': 0.03335042150831624, 'helicopter': 0.18073593073593075, 'small-vehicle': 0.1646635245920689, 'tennis-court': 0.8343676456645676, 'ground-track-field': 0.32647002944808584, 'bridge': 0.05187341394237946, 'storage-tank': 0.5208627143884783}, '0.9': {'baseball-diamond': 0.03305785123966942, 'mAP': 0.0839655052992035, 'basketball-court': 0.04467084639498432, 'ship': 0.022727272727272728, 'roundabout': 0.09090909090909091, 'plane': 0.11508224597709682, 'harbor': 0.004958677685950413, 'large-vehicle': 0.0404040404040404, 'soccer-ball-field': 0.0718475073313783, 'swimming-pool': 0.003305785123966942, 'helicopter': 0.045454545454545456, 'small-vehicle': 0.012396694214876032, 'tennis-court': 0.6362553328743662, 'ground-track-field': 0.045454545454545456, 'bridge': 0.0004784688995215311, 'storage-tank': 0.09247967479674796}, '0.85': {'baseball-diamond': 0.062229763012424055, 'mAP': 0.1913978363265622, 'basketball-court': 0.2929822574983865, 'ship': 0.15595653266793286, 'roundabout': 0.18268926195755464, 'plane': 0.36976262404457394, 'harbor': 0.09090909090909091, 'large-vehicle': 0.10402082575944191, 'soccer-ball-field': 0.18193971803597472, 'swimming-pool': 0.009828009828009828, 'helicopter': 0.09090909090909091, 'small-vehicle': 0.0507597361948061, 'tennis-court': 0.7870209211912281, 'ground-track-field': 0.16115702479338842, 'bridge': 0.011655011655011654, 'storage-tank': 0.3191476764415192}, '0.55': {'baseball-diamond': 0.6597077369398566, 'mAP': 0.6332910826026463, 'basketball-court': 0.635476033628063, 'ship': 0.8097646080629226, 'roundabout': 0.6568878949382035, 'plane': 0.8941072665170262, 'harbor': 0.5163443485924708, 'large-vehicle': 0.6059648816979448, 'soccer-ball-field': 0.6348976077949817, 'swimming-pool': 0.4870226878330306, 'helicopter': 0.44330776785866605, 'small-vehicle': 0.58803401943504, 'tennis-court': 0.9010162451888205, 'ground-track-field': 0.5471589944517752, 'bridge': 0.33952656587652713, 'storage-tank': 0.7801495802243646}, '0.95': {'baseball-diamond': 0.002932551319648094, 'mAP': 0.020975238643761322, 'basketball-court': 0.0, 'ship': 0.0005107252298263534, 'roundabout': 0.003367003367003367, 'plane': 0.009090909090909092, 'harbor': 0.0, 'large-vehicle': 0.00044267374944665776, 'soccer-ball-field': 0.0, 'swimming-pool': 0.0, 'helicopter': 0.045454545454545456, 'small-vehicle': 5.263989050902774e-05, 'tennis-court': 0.23979051856751876, 'ground-track-field': 0.0, 'bridge': 0.0, 'storage-tank': 0.012987012987012986}, 'mmAP': 0.3966134326781918, '0.7': {'baseball-diamond': 0.423911777372294, 'mAP': 0.4954183670446296, 'basketball-court': 0.5615948575117606, 'ship': 0.7015492582306313, 'roundabout': 0.5088098075188033, 'plane': 0.7920255473088503, 'harbor': 0.2347651397541761, 'large-vehicle': 0.44300772442305775, 'soccer-ball-field': 0.5136555705127277, 'swimming-pool': 0.25040329569326797, 'helicopter': 0.3228915703476295, 'small-vehicle': 0.4420316617271122, 'tennis-court': 0.8968498503736967, 'ground-track-field': 0.4435273770696969, 'bridge': 0.17134914681810218, 'storage-tank': 0.7249029210076382}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_2x_20210129'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "6,7,8,9z"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 20673 * 2

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
DECAY_STEP = [SAVE_WEIGHTS_INTE*18, SAVE_WEIGHTS_INTE*24, SAVE_WEIGHTS_INTE*30]
MAX_ITERATION = SAVE_WEIGHTS_INTE*30
WARM_SETP = int(1.0 / 8.0 * SAVE_WEIGHTS_INTE)

# -------------------------------------------- Dataset
DATASET_NAME = 'DOTATrain'  # 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 800
CLASS_NUM = 15

IMG_ROTATE = False
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# --------------------------------------------- Network
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 1e-4
USE_GN = False
FPN_CHANNEL = 256
NUM_SUBNET_CONV = 4
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
