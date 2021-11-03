# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

# gpu setting
GPU_GROUP = "0,1,2,3"

# log print
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
ADD_BOX_IN_TENSORBOARD = True

# learning policy
BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 5e-4
WEIGHT_DECAY = 1e-4
DECAY_EPOCH = [12, 16, 20]
MAX_EPOCH = 13
WARM_EPOCH = 1.0 / 4.0

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip
