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
SAVE_WEIGHTS_INTE = 5000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'UCAS-AOD'
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1500
CLASS_NUM = 2

# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CENTER_LOSS_MODE = 1  # center loss in kld
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.01

# test and eval
USE_07_METRIC = False

VERSION = 'RetinaNet_UCAS-AOD_KF_2x_20221106'


"""
RetinaNet-H + log(kl_center) + kfiou (exp(1-IoU)-1)

USE_07_METRIC
cls : car|| Recall: 0.977302204928664 || Precison: 0.3104016477857878|| AP: 0.8858512776730646
F1:0.9211993569498144 P:0.910126582278481 R:0.9325551232166018
cls : plane|| Recall: 0.9905191873589165 || Precison: 0.5553024550746647|| AP: 0.9050687140030588
F1:0.9788826278823866 P:0.9844748858447488 R:0.9733634311512416
mAP is : 0.8954599958380617

USE_12_METRIC
cls : car|| Recall: 0.977302204928664 || Precison: 0.31052956933855347|| AP: 0.945074330828179
F1:0.9212421878032977 P:0.9133205863607393 R:0.9293125810635539
cls : plane|| Recall: 0.9905191873589165 || Precison: 0.5553024550746647|| AP: 0.9841440144739153
F1:0.9788826278823866 P:0.9844748858447488 R:0.9733634311512416
mAP is : 0.9646091726510472
"""


