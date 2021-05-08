# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
RetinaNet-H + gwd fix bug + sqrt + tau=2 + train set
FLOPs: 860451163;    Trainable params: 33002916

iou threshold: 0.5
classname: plane
npos num: 2450
ap:  0.8948394008103565
classname: baseball-diamond
npos num: 209
ap:  0.6580467157774382
classname: bridge
npos num: 424
ap:  0.388917639526009
classname: ground-track-field
npos num: 131
ap:  0.582799082808811
classname: small-vehicle
npos num: 5090
ap:  0.6058372268499183
classname: large-vehicle
npos num: 4293
ap:  0.6297220646782561
classname: ship
npos num: 8861
ap:  0.8143495259256781
classname: tennis-court
npos num: 739
ap:  0.897082428301694
classname: basketball-court
npos num: 124
ap:  0.6194974348503025
classname: storage-tank
npos num: 1869
ap:  0.7888520103937031
classname: soccer-ball-field
npos num: 87
ap:  0.6721727619016967
classname: roundabout
npos num: 164
ap:  0.6740140076462648
classname: harbor
npos num: 2065
ap:  0.6030928319524497
classname: swimming-pool
npos num: 366
ap:  0.532690992577956
classname: helicopter
npos num: 72
ap:  0.45393048522054874
map: 0.6543896406147388

{'0.65': {'mAP': 0.5531255908346647, 'ground-track-field': 0.46874541967164557, 'small-vehicle': 0.5254805842312422, 'soccer-ball-field': 0.49674069740653076, 'harbor': 0.3325998985859663, 'large-vehicle': 0.49237446722103323, 'swimming-pool': 0.3786694115862947, 'roundabout': 0.6127737951332743, 'tennis-court': 0.8955950695702153, 'basketball-court': 0.5642336574393851, 'helicopter': 0.4095234559651532, 'storage-tank': 0.768350569402555, 'bridge': 0.229887299838382, 'baseball-diamond': 0.5172297968073052, 'ship': 0.718831628735693, 'plane': 0.885848110925295},
'0.5': {'mAP': 0.6543896406147388, 'ground-track-field': 0.582799082808811, 'small-vehicle': 0.6058372268499183, 'soccer-ball-field': 0.6721727619016967, 'harbor': 0.6030928319524497, 'large-vehicle': 0.6297220646782561, 'swimming-pool': 0.532690992577956, 'roundabout': 0.6740140076462648, 'tennis-court': 0.897082428301694, 'basketball-court': 0.6194974348503025, 'helicopter': 0.45393048522054874, 'storage-tank': 0.7888520103937031, 'bridge': 0.388917639526009, 'baseball-diamond': 0.6580467157774382, 'ship': 0.8143495259256781, 'plane': 0.8948394008103565},
'0.8': {'mAP': 0.28292248169049333, 'ground-track-field': 0.2325775080634852, 'small-vehicle': 0.1979511661753693, 'soccer-ball-field': 0.29786281543794524, 'harbor': 0.11494252873563218, 'large-vehicle': 0.16034195972421744, 'swimming-pool': 0.10212121212121213, 'roundabout': 0.29187883858274505, 'tennis-court': 0.8003975003061949, 'basketball-court': 0.47053242084058733, 'helicopter': 0.08282828282828283, 'storage-tank': 0.4630236938472425, 'bridge': 0.045454545454545456, 'baseball-diamond': 0.0980392156862745, 'ship': 0.3419243781838527, 'plane': 0.5439611593698137},
'0.85': {'mAP': 0.17732891599288997, 'ground-track-field': 0.13084951639168507, 'small-vehicle': 0.06282073067119796, 'soccer-ball-field': 0.18311688311688312, 'harbor': 0.09090909090909091, 'large-vehicle': 0.05997549072961212, 'swimming-pool': 0.01515151515151515, 'roundabout': 0.1523809523809524, 'tennis-court': 0.777850986366134, 'basketball-court': 0.27146743865010114, 'helicopter': 0.025974025974025972, 'storage-tank': 0.3194857000235097, 'bridge': 0.025974025974025972, 'baseball-diamond': 0.07032306536438768, 'ship': 0.09238611869237975, 'plane': 0.38126819949784874},
'0.9': {'mAP': 0.09261312239028942, 'ground-track-field': 0.045454545454545456, 'small-vehicle': 0.007575757575757575, 'soccer-ball-field': 0.08787878787878788, 'harbor': 0.09090909090909091, 'large-vehicle': 0.006888231631382316, 'swimming-pool': 0.01515151515151515, 'roundabout': 0.05694896083698572, 'tennis-court': 0.6190068314484273, 'basketball-court': 0.1277056277056277, 'helicopter': 0.018181818181818184, 'storage-tank': 0.10310064772905649, 'bridge': 0.012987012987012986, 'baseball-diamond': 0.05454545454545454, 'ship': 0.00899621212121212, 'plane': 0.133866341697667},
'0.6': {'mAP': 0.602003225559061, 'ground-track-field': 0.5117731722941454, 'small-vehicle': 0.5692796674261347, 'soccer-ball-field': 0.591601532425069, 'harbor': 0.42439117183385383, 'large-vehicle': 0.5379528999441402, 'swimming-pool': 0.4552774282858074, 'roundabout': 0.6590275695186874, 'tennis-court': 0.8967502975397331, 'basketball-court': 0.6163602294422292, 'helicopter': 0.42175379721391987, 'storage-tank': 0.7814590420239126, 'bridge': 0.30900189391187255, 'baseball-diamond': 0.6270284107602824, 'ship': 0.7357085211727478, 'plane': 0.892682749593379},
'0.7': {'mAP': 0.47209699491529994, 'ground-track-field': 0.37315990473910204, 'small-vehicle': 0.4462857945106512, 'soccer-ball-field': 0.43301958208470137, 'harbor': 0.24212265985665615, 'large-vehicle': 0.41707228898274396, 'swimming-pool': 0.2672845272755605, 'roundabout': 0.4752231061636024, 'tennis-court': 0.8954629342636613, 'basketball-court': 0.5565887540061711, 'helicopter': 0.3137137929820856, 'storage-tank': 0.6891634802537836, 'bridge': 0.16824841824841824, 'baseball-diamond': 0.3967626112242669, 'ship': 0.6233882592021442, 'plane': 0.7839588099359523},
'0.75': {'mAP': 0.38682933856456475, 'ground-track-field': 0.3505001362890805, 'small-vehicle': 0.32936925454926796, 'soccer-ball-field': 0.35644113950565565, 'harbor': 0.16082435022158342, 'large-vehicle': 0.312014321085313, 'swimming-pool': 0.15053744756715054, 'roundabout': 0.421342806894755, 'tennis-court': 0.8933998458347037, 'basketball-court': 0.5018426096266209, 'helicopter': 0.17586580086580086, 'storage-tank': 0.6481067305855587, 'bridge': 0.11431682090364725, 'baseball-diamond': 0.21312574893137554, 'ship': 0.5086325250920672, 'plane': 0.6661205405158923},
'mmAP': 0.38707336824937255,
'0.95': {'mAP': 0.020635306242343165, 'ground-track-field': 0.045454545454545456, 'small-vehicle': 0.0005790387955993052, 'soccer-ball-field': 0.0, 'harbor': 0.0004434589800443459, 'large-vehicle': 0.00036638424547744445, 'swimming-pool': 0.0, 'roundabout': 0.0053475935828877, 'tennis-court': 0.2304241077310939, 'basketball-court': 0.003189792663476874, 'helicopter': 0.0, 'storage-tank': 0.012987012987012986, 'bridge': 0.0, 'baseball-diamond': 0.0, 'ship': 0.0009404388714733542, 'plane': 0.009797220323536112},
'0.55': {'mAP': 0.6287890656893798, 'ground-track-field': 0.5643322633863954, 'small-vehicle': 0.5913067741856398, 'soccer-ball-field': 0.6335613572261539, 'harbor': 0.5190220297608497, 'large-vehicle': 0.5649195362143626, 'swimming-pool': 0.49227487366542605, 'roundabout': 0.667984152802187, 'tennis-court': 0.897082428301694, 'basketball-court': 0.6163602294422292, 'helicopter': 0.44399239228256077, 'storage-tank': 0.7862921590716214, 'bridge': 0.35810648582284893, 'baseball-diamond': 0.6568440654367499, 'ship': 0.7454706366368675, 'plane': 0.8942866011051104}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_2x_20210124'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,3"
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
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
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
