# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

DATASET_NAME = 'DOTA'
CLASS_NUM = 15
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 800

# data augmentation
IMG_ROTATE = False
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False
