# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 10000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'MLT'
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1000
CLASS_NUM = 1

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# post-processing
VIS_SCORE = 0.2

VERSION = 'RetinaNet_MLT_KF_1x_20210730'

"""
FLOPs: 472715469;    Trainable params: 32325246

2021-07-31  kf_0.4	55.96%	72.77%	45.46%	41.84%
2021-07-31  kf_0.45	55.59%	76.36%	43.70%	40.52%
2021-07-31  kl_0.35	55.78%	68.31%	47.13%	43.01%
2021-07-31  kf_0.3	54.96%	63.03%	48.72%	44.06%
2021-07-31	kf_0.25	53.16%	56.39%	50.28%	44.99%
"""

