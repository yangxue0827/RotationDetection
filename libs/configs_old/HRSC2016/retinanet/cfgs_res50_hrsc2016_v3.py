# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
RetinaNet-H

RetinaNet_HRSC2016_2x_20210210
FLOPs: 836056773;    Trainable params: 32325246

RetinaNet_HRSC2016_2x_20201127
loss inside weight (1, 1, 1, 1, 1)
cls : ship|| Recall: 0.9372964169381107 || Precison: 0.5391100702576113|| AP: 0.8427558835614876
F1:0.8743348342202211 P:0.8790123456790123 R:0.8697068403908795
mAP is : 0.8427558835614876

RetinaNet_HRSC2016_2x_20210210
loss inside weight (10, 1, 1, 1, 1)
cls : ship|| Recall: 0.9421824104234527 || Precison: 0.5565175565175565|| AP: 0.8456514504687931
F1:0.8766417085284619 P:0.8598277212216131 R:0.8941368078175895
mAP is : 0.8456514504687931

RetinaNet_HRSC2016_2x_20210210_v1
FLOPs: 836056773;    Trainable params: 32325246
loss inside weight (1, 10, 1, 1, 1)
cls : ship|| Recall: 0.9421824104234527 || Precison: 0.5741935483870968|| AP: 0.8467487586603686
F1:0.875045221879755 P:0.8636003172085647 R:0.8868078175895765
mAP is : 0.8467487586603686

RetinaNet_HRSC2016_2x_20210210_v2
FLOPs: 836056773;    Trainable params: 32325246
loss inside weight (1, 1, 10, 1, 1)
cls : ship|| Recall: 0.9283387622149837 || Precison: 0.32881453706374386|| AP: 0.8247113119468787
F1:0.847128760476153 P:0.8286604361370716 R:0.8664495114006515
mAP is : 0.8247113119468787

RetinaNet_HRSC2016_2x_20210210_v3
FLOPs: 836056773;    Trainable params: 32325246
loss inside weight (1, 1, 1, 10, 1)
cls : ship|| Recall: 0.9210097719869706 || Precison: 0.340765290750226|| AP: 0.8156900836170689
F1:0.8434171150322974 P:0.836 R:0.8509771986970684
mAP is : 0.8156900836170689

RetinaNet_HRSC2016_2x_20210210_v4
FLOPs: 836056773;    Trainable params: 32325246
loss inside weight (1, 1, 1, 1, 10)
cls : ship|| Recall: 0.9364820846905537 || Precison: 0.3406398104265403|| AP: 0.8193719796782251
F1:0.8465813463432883 P:0.8351822503961965 R:0.8583061889250815
mAP is : 0.8193719796782251
"""

# ------------------------------------------------
VERSION = 'RetinaNet_HRSC2016_2x_20210210_v4'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 10000 * 2

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
EVAL_THRESHOLD = 0.5
ADD_BOX_IN_TENSORBOARD = True

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 0.5
REG_LOSS_MODE = None
ALPHA = 1.0
BETA = 1.0

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 1e-3
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(1.0 / 4.0 * SAVE_WEIGHTS_INTE)

# -------------------------------------------- Dataset
DATASET_NAME = 'HRSC2016'  # 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 512
IMG_MAX_LENGTH = 512
CLASS_NUM = 1

IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
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
ANGLE_RANGE = 90  # 90 or 180

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


