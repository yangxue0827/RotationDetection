# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
R2CNN
FLOPs: 1024153266;    Trainable params: 41772682
This is your result for task 1:

    mAP: 0.7227414456963894
    ap of each class:
    plane:0.8954291131230108,
    baseball-diamond:0.7615013248230833,
    bridge:0.47589589239010427,
    ground-track-field:0.6484503831218632,
    small-vehicle:0.7616171143029637,
    large-vehicle:0.7395101403930869,
    ship:0.8587426481796258,
    tennis-court:0.9022025499507798,
    basketball-court:0.8327346869026073,
    storage-tank:0.8431585743608815,
    soccer-ball-field:0.5106006620292729,
    roundabout:0.6561468034665185,
    harbor:0.6530002955426998,
    swimming-pool:0.6823392612570894,
    helicopter:0.6197922356022552

The submitted information is :

Description: FPN_Res50D_DOTA_1x_20201031_37.8w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""

# ------------------------------------------------
VERSION = 'FPN_Res50D_DOTA_1x_20201031'
NET_NAME = 'resnet50_v1d'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 50
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 27000

SUMMARY_PATH = os.path.join(ROOT_PATH, 'output/summary')
TEST_SAVE_PATH = os.path.join(ROOT_PATH, 'tools/test_result')

pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_R_DIR = os.path.join(ROOT_PATH, 'output/evaluate_result_pickle/')

# ------------------------------------------ Train an test
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = False
FREEZE_BLOCKS = [True, True, False, False, False]  # for gluoncv backbone
FIXED_BLOCKS = 0  # allow 0~3
USE_07_METRIC = True
CUDA8 = False
ADD_BOX_IN_TENSORBOARD = True

RPN_LOCATION_LOSS_WEIGHT = 1.
RPN_CLASSIFICATION_LOSS_WEIGHT = 1.0
FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0
RPN_SIGMA = 3.0
FASTRCNN_SIGMA = 1.0

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.001 * BATCH_SIZE * NUM_GPU
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(1.0 / 4.0 * SAVE_WEIGHTS_INTE)

# -------------------------------------------- Dataset
DATASET_NAME = 'DOTA'  # 'ship', 'spacenet', 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 800
CLASS_NUM = 15

IMG_ROTATE = False
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# --------------------------------------------- Network
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.00004 if NET_NAME.startswith('Mobilenet') else 0.0001
ADD_GLOBAL_CTX = False
SHARE_HEADS = True
FPN_CHANNEL = 512

# --------------------------------------------- Anchor
USE_CENTER_OFFSET = False
LEVEL = ['P2', 'P3', 'P4', 'P5', 'P6']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]  # addjust the base anchor size for voc.
ANCHOR_STRIDE = [4, 8, 16, 32, 64]
ANCHOR_SCALES = [1.0]
ANCHOR_RATIOS = [0.5, 1., 2.0, 1/4.0, 4.0, 1/6.0, 6.0]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0, 2.0]
ANCHOR_SCALE_FACTORS = None
ANCHOR_MODE = 'H'
ANGLE_RANGE = 90

# -------------------------------------------- RPN
FPN_MODE = 'fpn'
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 512  # 256
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7  # 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000

RPN_TOP_K_NMS_TEST = 6000
RPN_MAXIMUM_PROPOSAL_TEST = 1000

# ------------------------------------------- Fast-RCNN
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 1.0
VIS_SCORE = 0.6  # only show in tensorboard
FILTERED_SCORE = 0.05

SOFT_NMS = False
FAST_RCNN_NMS_IOU_THRESHOLD = 0.3
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 200
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 512
FAST_RCNN_POSITIVE_RATE = 0.25
ADD_GTBOXES_TO_TRAIN = False
ROTATE_NMS_USE_GPU = True
