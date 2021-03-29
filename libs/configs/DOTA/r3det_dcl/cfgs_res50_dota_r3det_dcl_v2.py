# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2,3'
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 20673 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTATrain'

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_REFINE_STAGE = 1
ANGLE_RANGE = 180

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 0.5
USE_IOU_FACTOR = True

# DCL
OMEGA = 180 / 256.  
ANGLE_MODE = 0  # {0: BCL, 1: GCL}

VERSION = 'RetinaNet_DOTA_R3Det_DCL_B_2x_20210223'

"""
FLOPs: 1263438247;    Trainable params: 37785791
{'0.8': {'large-vehicle': 0.21404754951558433, 'harbor': 0.06153370439084725, 'plane': 0.5232719854619842, 'ground-track-field': 0.2720025072966249, 'mAP': 0.24564484377650206, 'basketball-court': 0.3581453654110614, 'storage-tank': 0.3654572933121789, 'soccer-ball-field': 0.37948418003565065, 'bridge': 0.027972027972027972, 'tennis-court': 0.7790032797975169, 'swimming-pool': 0.012316715542521995, 'small-vehicle': 0.0971913376221646, 'helicopter': 0.0303030303030303, 'ship': 0.15580302935334797, 'roundabout': 0.21695607763023494, 'baseball-diamond': 0.19118457300275482},
'0.9': {'large-vehicle': 0.007867132867132866, 'harbor': 0.0014354066985645933, 'plane': 0.05737300481715033, 'ground-track-field': 0.012987012987012986, 'mAP': 0.05075120803602134, 'basketball-court': 0.09090909090909091, 'storage-tank': 0.05920427119139911, 'soccer-ball-field': 0.11570247933884298, 'bridge': 0.006993006993006993, 'tennis-court': 0.29870225788821225, 'swimming-pool': 0.0014204545454545455, 'small-vehicle': 0.0020065771138732514, 'helicopter': 0.0, 'ship': 0.007736943907156673, 'roundabout': 0.008021390374331552, 'baseball-diamond': 0.09090909090909091},
'0.6': {'large-vehicle': 0.7178831860195826, 'harbor': 0.4118806613326167, 'plane': 0.8822473010673048, 'ground-track-field': 0.5282442141505738, 'mAP': 0.6049649936977721, 'basketball-court': 0.6323082326820151, 'storage-tank': 0.7867280261058349, 'soccer-ball-field': 0.5890374331550802, 'bridge': 0.29424698253763665, 'tennis-court': 0.8946914530293323, 'swimming-pool': 0.4098012474883804, 'small-vehicle': 0.5196961674671526, 'helicopter': 0.37322334276952107, 'ship': 0.7522526336197219, 'roundabout': 0.6206894014371582, 'baseball-diamond': 0.6615446226046721},
'0.75': {'large-vehicle': 0.3836145229865022, 'harbor': 0.1385983273295886, 'plane': 0.6637766636992023, 'ground-track-field': 0.38672595308730856, 'mAP': 0.3543632625214466, 'basketball-court': 0.47957357173330956, 'storage-tank': 0.5668597498391078, 'soccer-ball-field': 0.4417748917748918, 'bridge': 0.06519950637597696, 'tennis-court': 0.796780030974753, 'swimming-pool': 0.07126099706744868, 'small-vehicle': 0.18173104842933308, 'helicopter': 0.06932703659976387, 'ship': 0.35286510582112224, 'roundabout': 0.3944524260197862, 'baseball-diamond': 0.32290910608360496},
'mmAP': 0.3753593751816795,
'0.5': {'large-vehicle': 0.7465104034519473, 'harbor': 0.5564063785076355, 'plane': 0.8972576993499554, 'ground-track-field': 0.5579121336275416, 'mAP': 0.6744843322197934, 'basketball-court': 0.6559907006712523, 'storage-tank': 0.8575344684702998, 'soccer-ball-field': 0.674593403252774, 'bridge': 0.4072414360943429, 'tennis-court': 0.8952710406067274, 'swimming-pool': 0.543531128108074, 'small-vehicle': 0.6027910712294844, 'helicopter': 0.5181958198922268, 'ship': 0.8425248120301219, 'roundabout': 0.6400725949428725, 'baseball-diamond': 0.7214318930616466},
'0.7': {'large-vehicle': 0.5313126739462206, 'harbor': 0.21516785448240203, 'plane': 0.777734216426862, 'ground-track-field': 0.44607957731605896, 'mAP': 0.47499830304658514, 'basketball-court': 0.5879338524043383, 'storage-tank': 0.6811008443664961, 'soccer-ball-field': 0.5429283345791885, 'bridge': 0.16854000605387953, 'tennis-court': 0.8932669913290455, 'swimming-pool': 0.17915804002760524, 'small-vehicle': 0.33567153716500553, 'helicopter': 0.17510733541267895, 'ship': 0.5859517666151295, 'roundabout': 0.4831469479801222, 'baseball-diamond': 0.5218745675937457},
'0.85': {'large-vehicle': 0.045468039857067294, 'harbor': 0.022727272727272728, 'plane': 0.2762316600537012, 'ground-track-field': 0.11442006269592477, 'mAP': 0.1381517780714906, 'basketball-court': 0.2173862605180663, 'storage-tank': 0.18704403215010793, 'soccer-ball-field': 0.25329380764163373, 'bridge': 0.027972027972027972, 'tennis-court': 0.6513342645250559, 'swimming-pool': 0.006060606060606061, 'small-vehicle': 0.013824884792626729, 'helicopter': 0.022727272727272728, 'ship': 0.04844923594923595, 'roundabout': 0.06363636363636363, 'baseball-diamond': 0.1217008797653959},
'0.95': {'large-vehicle': 0.0006447453255963894, 'harbor': 0.0, 'plane': 0.0030243728873865857, 'ground-track-field': 0.0, 'mAP': 0.004232040139212309, 'basketball-court': 0.003189792663476874, 'storage-tank': 0.019886363636363636, 'soccer-ball-field': 0.0, 'bridge': 0.0008340283569641369, 'tennis-court': 0.011922503725782414, 'swimming-pool': 0.0, 'small-vehicle': 0.0009469696969696969, 'helicopter': 0.0, 'ship': 0.00030455306837216383, 'roundabout': 0.0, 'baseball-diamond': 0.022727272727272728},
'0.55': {'large-vehicle': 0.7353325878233784, 'harbor': 0.5211198528185972, 'plane': 0.8920862580585238, 'ground-track-field': 0.5407270672368901, 'mAP': 0.6488794667587067, 'basketball-court': 0.6559907006712523, 'storage-tank': 0.8355968791557072, 'soccer-ball-field': 0.6331860913730375, 'bridge': 0.36452763420441847, 'tennis-court': 0.8948645271848791, 'swimming-pool': 0.48934987486682935, 'small-vehicle': 0.5689992754728247, 'helicopter': 0.44766518942643774, 'ship': 0.8307465564549673, 'roundabout': 0.6261539298351151, 'baseball-diamond': 0.6968455767977416},
'0.65': {'large-vehicle': 0.6371479000855508, 'harbor': 0.31643099043398193, 'plane': 0.7939268425531657, 'ground-track-field': 0.5114564007421151, 'mAP': 0.557123523549265, 'basketball-court': 0.6254975167185005, 'storage-tank': 0.7777523498576263, 'soccer-ball-field': 0.5886586452762923, 'bridge': 0.22325398132542526, 'tennis-court': 0.8945475599023408, 'swimming-pool': 0.332120590973761, 'small-vehicle': 0.4495199055994137, 'helicopter': 0.2900912432210142, 'ship': 0.7227802850337782, 'roundabout': 0.5632017300383333, 'baseball-diamond': 0.6304669114776771}}
"""

