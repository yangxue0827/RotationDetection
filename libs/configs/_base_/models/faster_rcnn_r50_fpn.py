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
IS_FILTER_OUTSIDE_BOXES = False
FREEZE_BLOCKS = [True, True, False, False, False]  # for gluoncv backbone
FIXED_BLOCKS = 0  # allow 0~3

# neck
FPN_MODE = 'fpn'
SHARE_HEADS = True
FPN_CHANNEL = 512

# rpn head
LEVEL = ['P2', 'P3', 'P4', 'P5', 'P6']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [4, 8, 16, 32, 64]
ANCHOR_SCALES = [1.0]
ANCHOR_RATIOS = [0.5, 1., 2.0, 1/4.0, 4.0, 1/6.0, 6.0]
ANCHOR_ANGLES = [-90, -75, -60, -45, -30, -15]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0, 2.0]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = False
ANCHOR_MODE = 'H'
ANGLE_RANGE = 90

INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.00004 if NET_NAME.startswith('Mobilenet') else 0.0001
ADD_GLOBAL_CTX = False

# roi head
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 1.0

# loss
RPN_LOCATION_LOSS_WEIGHT = 1.
RPN_CLASSIFICATION_LOSS_WEIGHT = 1.0
FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0
RPN_SIGMA = 3.0
FASTRCNN_SIGMA = 1.0

# rpn sample
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 512
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000
RPN_TOP_K_NMS_TEST = 6000
RPN_MAXIMUM_PROPOSAL_TEST = 1000

# roi sample
CUDA8 = False  # assign level
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.0 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 512
FAST_RCNN_POSITIVE_RATE = 0.25
ADD_GTBOXES_TO_TRAIN = False

# post-processing
VIS_SCORE = 0.6
FILTERED_SCORE = 0.05
ROTATE_NMS_USE_GPU = True
SOFT_NMS = False
FAST_RCNN_NMS_IOU_THRESHOLD = 0.3
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 200

# test and eval
TEST_SAVE_PATH = os.path.join(ROOT_PATH, 'tools/test_result')
EVALUATE_R_DIR = os.path.join(ROOT_PATH, 'output/evaluate_result_pickle/')
USE_07_METRIC = True
EVAL_THRESHOLD = 0.5
