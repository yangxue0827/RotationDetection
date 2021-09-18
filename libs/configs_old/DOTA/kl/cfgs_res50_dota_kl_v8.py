# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
RetinaNet-H + kl (fix bug) + sqrt + tau=2
{'0.9': {'ship': 0.09090909090909091, 'small-vehicle': 0.015408320493066256, 'bridge': 0.01515151515151515, 'swimming-pool': 0.0025020850708924107, 'tennis-court': 0.6454933258211097, 'ground-track-field': 0.045454545454545456, 'basketball-court': 0.07262365810572088, 'storage-tank': 0.11397871607795272, 'roundabout': 0.09090909090909091, 'soccer-ball-field': 0.1443850267379679, 'large-vehicle': 0.010652898503365793, 'helicopter': 0.006993006993006993, 'plane': 0.15864726834713788, 'baseball-diamond': 0.045454545454545456, 'harbor': 0.011857707509881422, 'mAP': 0.09802805343592597},
'0.6': {'ship': 0.8427582781220939, 'small-vehicle': 0.6382446727413831, 'bridge': 0.27888758773604627, 'swimming-pool': 0.410406778753221, 'tennis-court': 0.8955142096115858, 'ground-track-field': 0.5626054865797375, 'basketball-court': 0.6126679985227477, 'storage-tank': 0.7863454998294311, 'roundabout': 0.6373171398267504, 'soccer-ball-field': 0.6940432703336128, 'large-vehicle': 0.6968569946991302, 'helicopter': 0.5486332851909186, 'plane': 0.8965784513533464, 'baseball-diamond': 0.6052721332757642, 'harbor': 0.5100663832772241, 'mAP': 0.6410798779901994},
'0.65': {'ship': 0.8227804677872368, 'small-vehicle': 0.6044503533649042, 'bridge': 0.22096129337211143, 'swimming-pool': 0.30492941552267533, 'tennis-court': 0.8934776228501253, 'ground-track-field': 0.543919500212072, 'basketball-court': 0.6106343019106792, 'storage-tank': 0.7750027925321344, 'roundabout': 0.601162673297683, 'soccer-ball-field': 0.6509582857839864, 'large-vehicle': 0.6736680639663845, 'helicopter': 0.49842582248834194, 'plane': 0.8945173597990239, 'baseball-diamond': 0.5630996815280283, 'harbor': 0.42009456076854373, 'mAP': 0.6052054796789287},
'0.7': {'ship': 0.7490858534390797, 'small-vehicle': 0.5320481618653765, 'bridge': 0.1386714572257462, 'swimming-pool': 0.20679394707785034, 'tennis-court': 0.8933476337396322, 'ground-track-field': 0.48860401092199973, 'basketball-court': 0.6026169116644173, 'storage-tank': 0.7433899339066385, 'roundabout': 0.5042054029758948, 'soccer-ball-field': 0.5123732999348762, 'large-vehicle': 0.5945904123397716, 'helicopter': 0.3789298120626715, 'plane': 0.822118181083594, 'baseball-diamond': 0.4075632599730201, 'harbor': 0.3176563830614355, 'mAP': 0.5261329774181337},
'0.5': {'ship': 0.854076369964659, 'small-vehicle': 0.6714964348397323, 'bridge': 0.36804769455204694, 'swimming-pool': 0.522163163485748, 'tennis-court': 0.8966542167216246, 'ground-track-field': 0.6197771687947338, 'basketball-court': 0.6126679985227477, 'storage-tank': 0.825384288358113, 'roundabout': 0.6508188176056385, 'soccer-ball-field': 0.7057305505130766, 'large-vehicle': 0.7393194356263213, 'helicopter': 0.550994898353162, 'plane': 0.8980334265992302, 'baseball-diamond': 0.660715849253786, 'harbor': 0.6456734790514447, 'mAP': 0.6814369194828044},
'0.75': {'ship': 0.6752729779956379, 'small-vehicle': 0.41961025256076484, 'bridge': 0.07115613499541364, 'swimming-pool': 0.12179573990597613, 'tennis-court': 0.8911662398237076, 'ground-track-field': 0.42587167943987125, 'basketball-court': 0.5634041401044474, 'storage-tank': 0.6565200300659023, 'roundabout': 0.40172861621137484, 'soccer-ball-field': 0.451738640763031, 'large-vehicle': 0.48508961536424483, 'helicopter': 0.2952302952302952, 'plane': 0.7520834380647851, 'baseball-diamond': 0.2339523298668805, 'harbor': 0.22664789125426732, 'mAP': 0.44475120144310665},
'0.8': {'ship': 0.48195834579482255, 'small-vehicle': 0.23054861474074378, 'bridge': 0.03674834682652493, 'swimming-pool': 0.05736677115987461, 'tennis-court': 0.8855762171254655, 'ground-track-field': 0.33158033189381153, 'basketball-court': 0.45882197990688556, 'storage-tank': 0.5374912625902607, 'roundabout': 0.23048849047034167, 'soccer-ball-field': 0.3437157296130061, 'large-vehicle': 0.33911783041763727, 'helicopter': 0.1315913815913816, 'plane': 0.6218624025990389, 'baseball-diamond': 0.13744297032706737, 'harbor': 0.12412395161651782, 'mAP': 0.3298956417782254},
'0.55': {'ship': 0.8483141767659046, 'small-vehicle': 0.6631208289632391, 'bridge': 0.3421979178634005, 'swimming-pool': 0.47796631839029313, 'tennis-court': 0.8966542167216246, 'ground-track-field': 0.6147956815111073, 'basketball-court': 0.6126679985227477, 'storage-tank': 0.811081767132927, 'roundabout': 0.647631458891496, 'soccer-ball-field': 0.6960281299504782, 'large-vehicle': 0.7116374861314491, 'helicopter': 0.550994898353162, 'plane': 0.8975967276037711, 'baseball-diamond': 0.6446206307160344, 'harbor': 0.591411214542773, 'mAP': 0.6671146301373605},
'0.95': {'ship': 0.0023923444976076554, 'small-vehicle': 0.0008045052292839903, 'bridge': 0.0, 'swimming-pool': 0.0, 'tennis-court': 0.2237637569727122, 'ground-track-field': 0.011363636363636364, 'basketball-court': 0.006818181818181818, 'storage-tank': 0.013986013986013986, 'roundabout': 0.0, 'soccer-ball-field': 0.0, 'large-vehicle': 0.0013443118803562427, 'helicopter': 0.0, 'plane': 0.022727272727272728, 'baseball-diamond': 0.006493506493506493, 'harbor': 0.0, 'mAP': 0.019312901997904766},
'mmAP': 0.4214779863460743,
'0.85': {'ship': 0.20836929917722288, 'small-vehicle': 0.07665806612320987, 'bridge': 0.0303030303030303, 'swimming-pool': 0.045454545454545456, 'tennis-court': 0.7827268278896572, 'ground-track-field': 0.1853246753246753, 'basketball-court': 0.2753986051898766, 'storage-tank': 0.33730738462101384, 'roundabout': 0.13125592356361587, 'soccer-ball-field': 0.2597402597402597, 'large-vehicle': 0.095714917046135, 'helicopter': 0.09090909090909091, 'plane': 0.38850953082214973, 'baseball-diamond': 0.07261292166952543, 'harbor': 0.04704762363829172, 'mAP': 0.20182218009815328}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_KL_1x_20210202'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 20673

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

# -------------------------------------------- KL
KL_TAU = 2.0
KL_FUNC = 0   # 0: sqrt   1: log
