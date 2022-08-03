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
SAVE_WEIGHTS_INTE = 20673
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_EPOCH = 1. / 8.
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTATrain'

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA_BCD_1x_20210719_v2'

"""
RetinaNet-H + bcd
FLOPs: 484911766;    Trainable params: 33002916

1-1/(sqrt(bcd)+2)
AP50:95: [0.6782728325015103, 0.6662735173751326, 0.6419947352854566, 0.5970469238117249, 0.5285683841450383,
          0.4240009416930086, 0.3105312939641359, 0.18192094547101353, 0.07956613493202704, 0.015335656943791442]
mmAP: 0.41235113661228384
++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--
{'0.65': {'swimming-pool': 0.3629991687249472, 'large-vehicle': 0.6616845198086584, 'bridge': 0.26824830114112713, 'mAP': 0.5970469238117249, 'harbor': 0.41660111065439226, 'small-vehicle': 0.5747117730127902, 'ship': 0.8069244361642545, 'soccer-ball-field': 0.6278770526178534, 'tennis-court': 0.8989770540107346, 'basketball-court': 0.5717924302933374, 'helicopter': 0.5101798628381191, 'roundabout': 0.5579745128044814, 'baseball-diamond': 0.5334812216203212, 'ground-track-field': 0.5067784837495467, 'storage-tank': 0.7642311143062476, 'plane': 0.8932428154290606}, '0.75': {'swimming-pool': 0.14172077922077922, 'large-vehicle': 0.4783790095487047, 'bridge': 0.10613415710503088, 'mAP': 0.4240009416930086, 'harbor': 0.22097678499195417, 'small-vehicle': 0.3549866654001863, 'ship': 0.6215448480205183, 'soccer-ball-field': 0.5006612282145454, 'tennis-court': 0.8969977021049543, 'basketball-court': 0.480545384059631, 'helicopter': 0.24748130657221565, 'roundabout': 0.3779510448945665, 'baseball-diamond': 0.17782968020122172, 'ground-track-field': 0.4156360354897441, 'storage-tank': 0.5752583279932618, 'plane': 0.7639111715778166}, '0.85': {'swimming-pool': 0.005145797598627788, 'large-vehicle': 0.15024837333510296, 'bridge': 0.012396694214876032, 'mAP': 0.18192094547101353, 'harbor': 0.09090909090909091, 'small-vehicle': 0.05752986963254398, 'ship': 0.1563863955168786, 'soccer-ball-field': 0.23458110516934047, 'tennis-court': 0.7974314996514289, 'basketball-court': 0.19506827164929932, 'helicopter': 0.006060606060606061, 'roundabout': 0.13593073593073593, 'baseball-diamond': 0.1034688995215311, 'ground-track-field': 0.20871899185152198, 'storage-tank': 0.1893827191485089, 'plane': 0.3855551318751104}, '0.55': {'swimming-pool': 0.4722732711428096, 'large-vehicle': 0.7122897676949733, 'bridge': 0.3640167700439652, 'mAP': 0.6662735173751326, 'harbor': 0.618749088227424, 'small-vehicle': 0.6266666134719314, 'ship': 0.8477281821090316, 'soccer-ball-field': 0.7061671149983197, 'tennis-court': 0.8989770540107346, 'basketball-court': 0.603984184269993, 'helicopter': 0.5753062733794313, 'roundabout': 0.6246096855599383, 'baseball-diamond': 0.6737809068633673, 'ground-track-field': 0.561305526709816, 'storage-tank': 0.8112487499071765, 'plane': 0.896999572238077}, '0.7': {'swimming-pool': 0.2613178154998984, 'large-vehicle': 0.5972156457221613, 'bridge': 0.1795607526135271, 'mAP': 0.5285683841450383, 'harbor': 0.31443314578704984, 'small-vehicle': 0.5017100586809958, 'ship': 0.7379075558329113, 'soccer-ball-field': 0.6013325814777993, 'tennis-court': 0.8988445334700508, 'basketball-court': 0.5717924302933374, 'helicopter': 0.3978975231518025, 'roundabout': 0.48272720515874623, 'baseball-diamond': 0.3412079058662704, 'ground-track-field': 0.47193808451696667, 'storage-tank': 0.6957634201123848, 'plane': 0.874877103991671}, '0.6': {'swimming-pool': 0.44882587978739297, 'large-vehicle': 0.7016568490215807, 'bridge': 0.3214156995187303, 'mAP': 0.6419947352854566, 'harbor': 0.510833228324236, 'small-vehicle': 0.6101095134728701, 'ship': 0.8341651748464334, 'soccer-ball-field': 0.6977076016713326, 'tennis-court': 0.8989770540107346, 'basketball-court': 0.6034471964070774, 'helicopter': 0.5708132916484605, 'roundabout': 0.6132946792204965, 'baseball-diamond': 0.6209297748158772, 'ground-track-field': 0.520825621805481, 'storage-tank': 0.7806820117523007, 'plane': 0.8962374529788442}, '0.8': {'swimming-pool': 0.10545454545454545, 'large-vehicle': 0.3245744766259589, 'bridge': 0.045454545454545456, 'mAP': 0.3105312939641359, 'harbor': 0.14359355146111524, 'small-vehicle': 0.18317079420270807, 'ship': 0.44506925950438525, 'soccer-ball-field': 0.3425584351068703, 'tennis-court': 0.8872769599016807, 'basketball-court': 0.3739152084222506, 'helicopter': 0.06099960645415191, 'roundabout': 0.22762412738253981, 'baseball-diamond': 0.12834224598930483, 'ground-track-field': 0.3304575748794816, 'storage-tank': 0.4493758572344647, 'plane': 0.6101022213880367}, '0.9': {'swimming-pool': 0.0013468013468013469, 'large-vehicle': 0.010145434554450496, 'bridge': 0.002066115702479339, 'mAP': 0.07956613493202704, 'harbor': 0.022727272727272728, 'small-vehicle': 0.005526577247888723, 'ship': 0.014545454545454545, 'soccer-ball-field': 0.12207792207792208, 'tennis-court': 0.6692993718817165, 'basketball-court': 0.04117259552042161, 'helicopter': 0.00505050505050505, 'roundabout': 0.005681818181818182, 'baseball-diamond': 0.045454545454545456, 'ground-track-field': 0.10160427807486631, 'storage-tank': 0.0303030303030303, 'plane': 0.11649030131123284}, '0.95': {'swimming-pool': 0.0, 'large-vehicle': 0.0004502679094060966, 'bridge': 0.0, 'mAP': 0.015335656943791442, 'harbor': 6.163328197226503e-05, 'small-vehicle': 0.0003551136363636364, 'ship': 0.0009090909090909091, 'soccer-ball-field': 0.02727272727272727, 'tennis-court': 0.18787413303542336, 'basketball-court': 0.0, 'helicopter': 0.0, 'roundabout': 0.0, 'baseball-diamond': 0.0, 'ground-track-field': 0.007575757575757575, 'storage-tank': 0.0, 'plane': 0.005536130536130536}, 'mmAP': 0.41235113661228384, '0.5': {'swimming-pool': 0.5119809144727974, 'large-vehicle': 0.7242373212293386, 'bridge': 0.3753703547167131, 'mAP': 0.6782728325015103, 'harbor': 0.6585099296217831, 'small-vehicle': 0.6338491353172088, 'ship': 0.8523959524229934, 'soccer-ball-field': 0.7233003867315855, 'tennis-court': 0.8989770540107346, 'basketball-court': 0.6071230020435936, 'helicopter': 0.5760987910778925, 'roundabout': 0.6421763343371174, 'baseball-diamond': 0.6863299399361775, 'ground-track-field': 0.5705881671156733, 'storage-tank': 0.8161538287125205, 'plane': 0.8970013757765242}}

sqrt(bcd)

"""
