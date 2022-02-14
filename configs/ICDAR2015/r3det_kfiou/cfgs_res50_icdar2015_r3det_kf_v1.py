# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0'
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 10000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'ICDAR2015'
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1000
CLASS_NUM = 1

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# post-processing
VIS_SCORE = 0.7

VERSION = 'RetinaNet_ICDAR2015_R3Det_KF_1x_20210731'

"""
0.7: Calculated!{"precision": 0.8006396588486141, "recall": 0.7231584015406837, "hmean": 0.7599291677207184, "AP": 0}%
0.68: Calculated!{"precision": 0.7927597061909759, "recall": 0.7274915743861339, "hmean": 0.7587245794627167, "AP": 0}%
0.72: Calculated!{"precision": 0.808695652173913, "recall": 0.7164179104477612, "hmean": 0.7597651263722236, "AP": 0}%
0.75: Calculated!{"precision": 0.8173719376391982, "recall": 0.7067886374578719, "hmean": 0.7580686806093467, "AP": 0}%
0.8: Calculated!{"precision": 0.8347876672484003, "recall": 0.6909003370245547, "hmean": 0.756059009483667, "AP": 0}%
0.85: Calculated!{"precision": 0.8577639751552795, "recall": 0.6649012999518537, "hmean": 0.7491185245457012, "AP": 0}%
0.92: Calculated!{"precision": 0.9009332376166547, "recall": 0.6042368801155513, "hmean": 0.7233429394812682, "AP": 0}%

Hmean50:95: 0.7599291677207184  0.7265368074879838  0.6804958259549709  0.6000505944852013  0.4973437895269415
            0.34454844421958003  0.1846698709840627  0.07133822413356945  0.011130786744244878  0.0005059448520111307
"""

