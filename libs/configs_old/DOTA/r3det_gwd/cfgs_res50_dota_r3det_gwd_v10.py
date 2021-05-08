# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
r3det+gwd (only refine stage) + sqrt tau=2
{'0.65': {'soccer-ball-field': 0.6476708563573841, 'basketball-court': 0.6289794162360223, 'helicopter': 0.3677521249728531, 'mAP': 0.5957332831137113, 'large-vehicle': 0.7413667039853573, 'ground-track-field': 0.525150063828202, 'ship': 0.7640522895056856, 'tennis-court': 0.8970750657639832, 'roundabout': 0.592308497103204, 'bridge': 0.2611836497011585, 'baseball-diamond': 0.5831428586744279, 'harbor': 0.3381509114015628, 'small-vehicle': 0.5415327991966344, 'plane': 0.8922116493769559, 'storage-tank': 0.7789127674587428, 'swimming-pool': 0.3765095931434964},
'0.95': {'soccer-ball-field': 0.011363636363636364, 'basketball-court': 0.0018181818181818182, 'helicopter': 0.0, 'mAP': 0.014581417237857087, 'large-vehicle': 0.0101010101010101, 'ground-track-field': 0.0010570824524312897, 'ship': 0.0003040437823046519, 'tennis-court': 0.17040598132171553, 'roundabout': 0.012987012987012986, 'bridge': 0.0, 'baseball-diamond': 0.0, 'harbor': 9.519276534983343e-05, 'small-vehicle': 8.080808080808081e-05, 'plane': 0.002932551319648094, 'storage-tank': 0.007575757575757575, 'swimming-pool': 0.0},
'0.5': {'soccer-ball-field': 0.7202035777335448, 'basketball-court': 0.6570358290595782, 'helicopter': 0.507359668590544, 'mAP': 0.6927513533278032, 'large-vehicle': 0.8075585850998929, 'ground-track-field': 0.5847808673597761, 'ship': 0.8619666181676511, 'tennis-court': 0.9012961606825425, 'roundabout': 0.665919570488276, 'bridge': 0.4045876475232275, 'baseball-diamond': 0.6749007400489488, 'harbor': 0.623834467636632, 'small-vehicle': 0.6788232224338521, 'plane': 0.8964828794079398, 'storage-tank': 0.8557000843518636, 'swimming-pool': 0.5508203813327787},
'0.6': {'soccer-ball-field': 0.6941918656704202, 'basketball-court': 0.6440291233802058, 'helicopter': 0.4529215049846265, 'mAP': 0.637498497961135, 'large-vehicle': 0.7528499043728778, 'ground-track-field': 0.5417902126157572, 'ship': 0.845067956906576, 'tennis-court': 0.899975147780768, 'roundabout': 0.6066704080833966, 'bridge': 0.30159189323895746, 'baseball-diamond': 0.6088660523652101, 'harbor': 0.43108000466397656, 'small-vehicle': 0.6168937718623935, 'plane': 0.8955821822347066, 'storage-tank': 0.8149692831796187, 'swimming-pool': 0.45599815807753485},
'mmAP': 0.41558145915711925,
'0.55': {'soccer-ball-field': 0.7074202814834487, 'basketball-court': 0.6440291233802058, 'helicopter': 0.4753343491455896, 'mAP': 0.6687019556583548, 'large-vehicle': 0.7893291399905034, 'ground-track-field': 0.5621063402393325, 'ship': 0.8551118409539427, 'tennis-court': 0.9011648202783585, 'roundabout': 0.6487529479493018, 'bridge': 0.35987633074598063, 'baseball-diamond': 0.6573202307345517, 'harbor': 0.5288572303943255, 'small-vehicle': 0.6468852035146908, 'plane': 0.8959976151749034, 'storage-tank': 0.8447622846434745, 'swimming-pool': 0.513581596246712},
'0.85': {'soccer-ball-field': 0.3652761795166859, 'basketball-court': 0.19763526740270926, 'helicopter': 0.0303030303030303, 'mAP': 0.19360541415868818, 'large-vehicle': 0.16071412214006728, 'ground-track-field': 0.19735323009247843, 'ship': 0.09763388133374654, 'tennis-court': 0.7715364478085396, 'roundabout': 0.16150636802810717, 'bridge': 0.045454545454545456, 'baseball-diamond': 0.11527647610121837, 'harbor': 0.022518765638031693, 'small-vehicle': 0.05808356490670902, 'plane': 0.3362311319343998, 'storage-tank': 0.2991036562655085, 'swimming-pool': 0.045454545454545456},
'0.8': {'soccer-ball-field': 0.47187714481811943, 'basketball-court': 0.43041655075073487, 'helicopter': 0.11666666666666667, 'mAP': 0.31073560656891364, 'large-vehicle': 0.34199986229619006, 'ground-track-field': 0.2786561264822134, 'ship': 0.3379979684121435, 'tennis-court': 0.7975125385388567, 'roundabout': 0.2382634745677015, 'bridge': 0.07376798285889194, 'baseball-diamond': 0.21058672021753883, 'harbor': 0.0670846394984326, 'small-vehicle': 0.1433581263559504, 'plane': 0.6255685918234501, 'storage-tank': 0.467584386517351, 'swimming-pool': 0.05969331872946331},
'0.7': {'soccer-ball-field': 0.6329046550785681, 'basketball-court': 0.6079021540987147, 'helicopter': 0.30877702710586813, 'mAP': 0.5252345295554081, 'large-vehicle': 0.6497653242856736, 'ground-track-field': 0.47582552394991745, 'ship': 0.728360173665124, 'tennis-court': 0.8933173370769347, 'roundabout': 0.49953326257440334, 'bridge': 0.1812955796067406, 'baseball-diamond': 0.47623331150141907, 'harbor': 0.24194745561512673, 'small-vehicle': 0.4344047121532568, 'plane': 0.7978135313507922, 'storage-tank': 0.7450217075641696, 'swimming-pool': 0.2054161877044136},
'0.75': {'soccer-ball-field': 0.6101426905881541, 'basketball-court': 0.5505935597263579, 'helicopter': 0.17243588160047085, 'mAP': 0.4334615485571184, 'large-vehicle': 0.5280125522007204, 'ground-track-field': 0.40087300600296494, 'ship': 0.5885175764907138, 'tennis-court': 0.8849137323269342, 'roundabout': 0.39203666807250237, 'bridge': 0.10729860251391352, 'baseball-diamond': 0.32264315373748836, 'harbor': 0.14245358654551138, 'small-vehicle': 0.30730191031397946, 'plane': 0.7672167651302257, 'storage-tank': 0.6285738129262044, 'swimming-pool': 0.0989097301806332},
'0.9': {'soccer-ball-field': 0.11363636363636363, 'basketball-court': 0.05726092089728453, 'helicopter': 0.0303030303030303, 'mAP': 0.08351098543220324, 'large-vehicle': 0.0303030303030303, 'ground-track-field': 0.06025369978858351, 'ship': 0.0106951871657754, 'tennis-court': 0.5426011157691901, 'roundabout': 0.045454545454545456, 'bridge': 0.018181818181818184, 'baseball-diamond': 0.09090909090909091, 'harbor': 0.01515151515151515, 'small-vehicle': 0.045454545454545456, 'plane': 0.12164390014130946, 'storage-tank': 0.06965547674089295, 'swimming-pool': 0.0011605415860735009}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_R3Det_GWD_2x_20210223'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2,3"
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
