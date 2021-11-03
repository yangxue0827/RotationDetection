# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import math

ROOT_PATH = os.path.abspath('../../')
print(ROOT_PATH)
SUMMARY_PATH = os.path.join(ROOT_PATH, 'output/summary')

# backbone
NET_NAME = 'resnet50_v1d'
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone

# neck
FPN_MODE = 'fpn'
SHARE_NET = True
USE_P5 = True
FPN_CHANNEL = 256

# bbox head
NUM_SUBNET_CONV = 4
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [1, 1 / 2, 2., 1 / 3., 3., 5., 1 / 5.]
ANCHOR_ANGLES = [-90, -75, -60, -45, -30, -15]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True
METHOD = 'H'
ANGLE_RANGE = 90  # 90 or 180
USE_GN = False

SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0

# sample
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4

# post-processing
NMS = True
NMS_IOU_THRESHOLD = 0.3
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.4

# test and eval
TEST_SAVE_PATH = os.path.join(ROOT_PATH, 'tools/test_result')
EVALUATE_R_DIR = os.path.join(ROOT_PATH, 'output/evaluate_result_pickle/')
USE_07_METRIC = True
EVAL_THRESHOLD = 0.5
