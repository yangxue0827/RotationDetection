# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
{'mmAP': 0.3846353435002476,
'0.55': {'harbor': 0.45492403380153207, 'mAP': 0.6464644172917693, 'plane': 0.8933446945455119, 'baseball-diamond': 0.6643879036044499, 'basketball-court': 0.6285957764960233, 'roundabout': 0.6104353673095043, 'soccer-ball-field': 0.6732276932312491, 'bridge': 0.32565802133268307, 'small-vehicle': 0.6192759548858993, 'storage-tank': 0.82560991199433, 'ground-track-field': 0.5741948847350586, 'tennis-court': 0.9025901584329094, 'large-vehicle': 0.7355124509304517, 'ship': 0.8570920383019739, 'swimming-pool': 0.5090798813564961, 'helicopter': 0.42303748841846756},
'0.8': {'harbor': 0.0404040404040404, 'mAP': 0.27566244982765425, 'plane': 0.5629049341944605, 'baseball-diamond': 0.20247561578206738, 'basketball-court': 0.4414830998122201, 'roundabout': 0.25517241379310346, 'soccer-ball-field': 0.35371267150928165, 'bridge': 0.025974025974025972, 'small-vehicle': 0.15135722883314345, 'storage-tank': 0.4431670880072948, 'ground-track-field': 0.24675591020956072, 'tennis-court': 0.7913582305331757, 'large-vehicle': 0.17029054611818153, 'ship': 0.25399349635681323, 'swimming-pool': 0.10497835497835498, 'helicopter': 0.09090909090909091},
'0.7': {'harbor': 0.1934530319177369, 'mAP': 0.47771024210530527, 'plane': 0.7973373945000658, 'baseball-diamond': 0.4124870902111155, 'basketball-court': 0.593781931167414, 'roundabout': 0.4788600934545909, 'soccer-ball-field': 0.5414718882768419, 'bridge': 0.15092003753127606, 'small-vehicle': 0.3994803158258394, 'storage-tank': 0.6784387643078624, 'ground-track-field': 0.42573953483044397, 'tennis-court': 0.9004282790888039, 'large-vehicle': 0.522261288594104, 'ship': 0.6444599654312124, 'swimming-pool': 0.2250188649271218, 'helicopter': 0.2015151515151515},
'0.95': {'harbor': 0.0003756574004507889, 'mAP': 0.008230590874466415, 'plane': 0.002331002331002331, 'baseball-diamond': 0.0011223344556677889, 'basketball-court': 0.0, 'roundabout': 0.022727272727272728, 'soccer-ball-field': 0.0, 'bridge': 0.0, 'small-vehicle': 0.0, 'storage-tank': 0.005681818181818182, 'ground-track-field': 0.0, 'tennis-court': 0.09090909090909091, 'large-vehicle': 0.00020337604230221678, 'ship': 0.0001083110693912918, 'swimming-pool': 0.0, 'helicopter': 0.0},
'0.65': {'harbor': 0.29761599568517544, 'mAP': 0.5592939480497944, 'plane': 0.8848950557089915, 'baseball-diamond': 0.5280637786776091, 'basketball-court': 0.6135513705063803, 'roundabout': 0.5298388520565711, 'soccer-ball-field': 0.5835231718310645, 'bridge': 0.2210317537593879, 'small-vehicle': 0.49986954303593245, 'storage-tank': 0.7675460576342068, 'ground-track-field': 0.5284221761401995, 'tennis-court': 0.9012102247774185, 'large-vehicle': 0.6280234886338019, 'ship': 0.7526734228155597, 'swimming-pool': 0.37131214010235, 'helicopter': 0.28183218938226645},
'0.85': {'harbor': 0.011363636363636364, 'mAP': 0.15246455508604928, 'plane': 0.29185529360129964, 'baseball-diamond': 0.10431503979891077, 'basketball-court': 0.2430847308031774, 'roundabout': 0.12062937062937062, 'soccer-ball-field': 0.18060064935064937, 'bridge': 0.025974025974025972, 'small-vehicle': 0.025663069743847293, 'storage-tank': 0.2548926671718632, 'ground-track-field': 0.12530712530712532, 'tennis-court': 0.6745786074174169, 'large-vehicle': 0.0630012620608232, 'ship': 0.058264831539667945, 'swimming-pool': 0.01652892561983471, 'helicopter': 0.09090909090909091},
'0.6': {'harbor': 0.3969346216387887, 'mAP': 0.6126695309427872, 'plane': 0.8913964489848933, 'baseball-diamond': 0.6010013278700178, 'basketball-court': 0.6205016776124507, 'roundabout': 0.5690767757589962, 'soccer-ball-field': 0.6628084193227769, 'bridge': 0.2730890361875502, 'small-vehicle': 0.5607069274708069, 'storage-tank': 0.785140653020529, 'ground-track-field': 0.5537032207512677, 'tennis-court': 0.9019451382911625, 'large-vehicle': 0.7166721912459084, 'ship': 0.8094996148656806, 'swimming-pool': 0.47038863109140394, 'helicopter': 0.3771782800295734},
'0.9': {'harbor': 0.002932551319648094, 'mAP': 0.05788896501517006, 'plane': 0.04870277181983171, 'baseball-diamond': 0.03896103896103896, 'basketball-court': 0.0670995670995671, 'roundabout': 0.045454545454545456, 'soccer-ball-field': 0.05451062347614072, 'bridge': 0.012987012987012986, 'small-vehicle': 0.004434589800443459, 'storage-tank': 0.11682372505543237, 'ground-track-field': 0.013636363636363636, 'tennis-court': 0.31759090938226753, 'large-vehicle': 0.045454545454545456, 'ship': 0.007792207792207792, 'swimming-pool': 0.001044932079414838, 'helicopter': 0.09090909090909091},
'0.5': {'harbor': 0.541350595225887, 'mAP': 0.6719099381091576, 'plane': 0.89376090028101, 'baseball-diamond': 0.7010427402386605, 'basketball-court': 0.6285957764960233, 'roundabout': 0.645248730324319, 'soccer-ball-field': 0.6959989729173733, 'bridge': 0.3682130055525563, 'small-vehicle': 0.6412829861143257, 'storage-tank': 0.8528766332105531, 'ground-track-field': 0.5982862225425049, 'tennis-court': 0.9025901584329094, 'large-vehicle': 0.7452394431268583, 'ship': 0.8660265205004241, 'swimming-pool': 0.5670462204606407, 'helicopter': 0.43109016621331653},
'0.75': {'harbor': 0.08028065464774325, 'mAP': 0.38405879770032214, 'plane': 0.7392246316280731, 'baseball-diamond': 0.3185729175529652, 'basketball-court': 0.5097863656633352, 'roundabout': 0.3686558229706952, 'soccer-ball-field': 0.46426774806221455, 'bridge': 0.07196969696969696, 'small-vehicle': 0.27329347915831853, 'storage-tank': 0.5697425716291908, 'ground-track-field': 0.3505324126013781, 'tennis-court': 0.8911165846180953, 'large-vehicle': 0.38126155338032497, 'ship': 0.4928422166643412, 'swimming-pool': 0.13822419884734843, 'helicopter': 0.1111111111111111}}

"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_R3Det_2x_20210222'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2"
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
REG_WEIGHT = 1.0
USE_IOU_FACTOR = False

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 5e-4
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
USE_ANGLE_COND = False
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

