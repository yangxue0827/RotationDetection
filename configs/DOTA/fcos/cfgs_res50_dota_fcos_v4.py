# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 20673
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTATrain'

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
CTR_WEIGHT = 1.0
REG_LOSS_MODE = 0

VERSION = 'FCOS_DOTA_RSDet_1x_20210617'

"""
FCOS
FLOPs: 468484120;    Trainable params: 32090136

AP50:95: [0.6607414642462539, 0.6298756327623419, 0.5948472990582359, 0.547285798803981, 0.4774425270406064,
          0.38897426417953346, 0.2761496365662143, 0.16271387307571325, 0.07300790608116917, 0.014449568974715948]
mmAP: 0.3825487970788765
++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--
{'0.6': {'roundabout': 0.5996958877342894, 'harbor': 0.4053881513990886, 'swimming-pool': 0.4788518987512071, 'tennis-court': 0.90118765886208, 'ship': 0.7579200728241066, 'storage-tank': 0.7937462746987992, 'basketball-court': 0.5888684318482488, 'plane': 0.8860173492833375, 'small-vehicle': 0.5745958681773501, 'bridge': 0.2865868855379142, 'helicopter': 0.4079234473019154, 'baseball-diamond': 0.5688908554910672, 'large-vehicle': 0.5944516256099066, 'ground-track-field': 0.43876259027197784, 'mAP': 0.5948472990582359, 'soccer-ball-field': 0.6398224880822504}, '0.65': {'roundabout': 0.5322300781901355, 'harbor': 0.2932896838488139, 'swimming-pool': 0.36483371743340154, 'tennis-court': 0.8985189232316093, 'ship': 0.7325010953255912, 'storage-tank': 0.7777519040837776, 'basketball-court': 0.561090987279219, 'plane': 0.8745613751638394, 'small-vehicle': 0.520482323079476, 'bridge': 0.2347987848543271, 'helicopter': 0.34221609428221006, 'baseball-diamond': 0.5071769025372111, 'large-vehicle': 0.5684215746234316, 'ground-track-field': 0.4193465314478448, 'mAP': 0.547285798803981, 'soccer-ball-field': 0.5820670066788256}, '0.5': {'roundabout': 0.6701296184240721, 'harbor': 0.6103798347080517, 'swimming-pool': 0.5952000315623657, 'tennis-court': 0.90118765886208, 'ship': 0.8285987956084143, 'storage-tank': 0.8641474348133786, 'basketball-court': 0.6154163720445397, 'plane': 0.8906184607235158, 'small-vehicle': 0.6044791986600657, 'bridge': 0.38854362790557767, 'helicopter': 0.46889496556003046, 'baseball-diamond': 0.654493389687513, 'large-vehicle': 0.6752168472433339, 'ground-track-field': 0.4544648675650143, 'mAP': 0.6607414642462539, 'soccer-ball-field': 0.6893508603258567}, '0.55': {'roundabout': 0.6156714441993675, 'harbor': 0.5074600472460165, 'swimming-pool': 0.5419771757973524, 'tennis-court': 0.90118765886208, 'ship': 0.7688395891467831, 'storage-tank': 0.8580508738053514, 'basketball-court': 0.6065200904175767, 'plane': 0.8895436493181794, 'small-vehicle': 0.5959253517460688, 'bridge': 0.3445755236452594, 'helicopter': 0.46889496556003046, 'baseball-diamond': 0.5841998312574537, 'large-vehicle': 0.6424960867677294, 'ground-track-field': 0.44709911083239257, 'mAP': 0.6298756327623419, 'soccer-ball-field': 0.6756930928334871}, '0.7': {'roundabout': 0.4925392982066029, 'harbor': 0.20798134137834356, 'swimming-pool': 0.2651192758711077, 'tennis-court': 0.8957009990081705, 'ship': 0.6369066318973402, 'storage-tank': 0.7452122200547678, 'basketball-court': 0.4889659022519063, 'plane': 0.7894136339585797, 'small-vehicle': 0.4277099468507674, 'bridge': 0.1459466842716325, 'helicopter': 0.29671980911650336, 'baseball-diamond': 0.36190938937078926, 'large-vehicle': 0.4852581317048663, 'ground-track-field': 0.3628605651118396, 'mAP': 0.4774425270406064, 'soccer-ball-field': 0.5593940765558794}, '0.8': {'roundabout': 0.24753114197110973, 'harbor': 0.04728782486789604, 'swimming-pool': 0.03107195589311934, 'tennis-court': 0.7841111106299586, 'ship': 0.27601030747301813, 'storage-tank': 0.46253263811887174, 'basketball-court': 0.35233544532735905, 'plane': 0.6226540062200648, 'small-vehicle': 0.17472224086221383, 'bridge': 0.09090909090909091, 'helicopter': 0.047216349541930935, 'baseball-diamond': 0.1522656909653814, 'large-vehicle': 0.2108292135141926, 'ground-track-field': 0.17985845000770373, 'mAP': 0.2761496365662143, 'soccer-ball-field': 0.4629090821913031}, '0.95': {'roundabout': 0.022727272727272728, 'harbor': 0.0001851508979818552, 'swimming-pool': 0.0004614674665436086, 'tennis-court': 0.05551534452808338, 'ship': 0.0004662004662004662, 'storage-tank': 0.022727272727272728, 'basketball-court': 0.006269592476489028, 'plane': 0.0606060606060606, 'small-vehicle': 0.00040225261464199515, 'bridge': 0.0, 'helicopter': 0.0, 'baseball-diamond': 0.0013774104683195593, 'large-vehicle': 0.0005509641873278236, 'ground-track-field': 0.0, 'mAP': 0.014449568974715948, 'soccer-ball-field': 0.045454545454545456}, '0.85': {'roundabout': 0.11458120531154241, 'harbor': 0.010570824524312896, 'swimming-pool': 0.00494484594903005, 'tennis-court': 0.6504176318745679, 'ship': 0.09086319564074061, 'storage-tank': 0.2804996137439323, 'basketball-court': 0.23324376772652636, 'plane': 0.3591444470775287, 'small-vehicle': 0.1038238936487477, 'bridge': 0.09090909090909091, 'helicopter': 0.012987012987012986, 'baseball-diamond': 0.060958421423537704, 'large-vehicle': 0.06862639435253542, 'ground-track-field': 0.05793528505392913, 'mAP': 0.16271387307571325, 'soccer-ball-field': 0.301202465912664}, '0.9': {'roundabout': 0.06818181818181818, 'harbor': 0.007575757575757575, 'swimming-pool': 0.0009775171065493646, 'tennis-court': 0.40163301670885165, 'ship': 0.0101010101010101, 'storage-tank': 0.12567391708496004, 'basketball-court': 0.10645933014354067, 'plane': 0.11807954804366289, 'small-vehicle': 0.00507137490608565, 'bridge': 0.0101010101010101, 'helicopter': 0.006493506493506493, 'baseball-diamond': 0.022727272727272728, 'large-vehicle': 0.045454545454545456, 'ground-track-field': 0.00505050505050505, 'mAP': 0.07300790608116917, 'soccer-ball-field': 0.16153846153846152}, '0.75': {'roundabout': 0.3884233191051373, 'harbor': 0.11998928906145401, 'swimming-pool': 0.09122315592903829, 'tennis-court': 0.8870046137473057, 'ship': 0.5096195796405203, 'storage-tank': 0.6345947023300228, 'basketball-court': 0.4464158689510802, 'plane': 0.7632995986160181, 'small-vehicle': 0.305230534801535, 'bridge': 0.10507246376811594, 'helicopter': 0.201010101010101, 'baseball-diamond': 0.252338476204175, 'large-vehicle': 0.3816431480480776, 'ground-track-field': 0.25129176379176377, 'mAP': 0.38897426417953346, 'soccer-ball-field': 0.4974573476886573}, 'mmAP': 0.3825487970788765}
"""



