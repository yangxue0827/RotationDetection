# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 20673 * 2
DECAY_EPOCH = [18, 24, 30]
MAX_EPOCH = 25
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

VERSION = 'RetinaNet_DOTA_2x_20210126'

"""
RSDet-8p
FLOPs: 865678510;    Trainable params: 33148131
{'0.85': {'roundabout': 0.1610909090909091, 'ship': 0.11336680156410345, 'helicopter': 0.0213903743315508, 'swimming-pool': 0.004167752018754884, 'basketball-court': 0.2220573077715935, 'mAP': 0.17920576090056564, 'bridge': 0.0303030303030303, 'large-vehicle': 0.059585278299777875, 'tennis-court': 0.7806958595116096, 'storage-tank': 0.31291337193553775, 'plane': 0.4175900771834713, 'baseball-diamond': 0.11391018619934283, 'small-vehicle': 0.05009887232109455, 'ground-track-field': 0.11865045332026465, 'soccer-ball-field': 0.24893280632411066, 'harbor': 0.03333333333333333},
'0.8': {'roundabout': 0.28094915093252054, 'ship': 0.28370086913930453, 'helicopter': 0.1547911547911548, 'swimming-pool': 0.10315186246418338, 'basketball-court': 0.43356946174085875, 'mAP': 0.30924296519237354, 'bridge': 0.09568793065125583, 'large-vehicle': 0.17568365197636787, 'tennis-court': 0.8156060017452267, 'storage-tank': 0.46055887278167806, 'plane': 0.6529587671671762, 'baseball-diamond': 0.18123774413679505, 'small-vehicle': 0.16714345244558385, 'ground-track-field': 0.30215096798903174, 'soccer-ball-field': 0.4152861945614403, 'harbor': 0.11616839536302624},
'mmAP': 0.39633919415425495,
'0.75': {'roundabout': 0.4414459740237362, 'ship': 0.4912542327249696, 'helicopter': 0.24086101912188868, 'swimming-pool': 0.1402256874145268, 'basketball-court': 0.5289330456532144, 'mAP': 0.4258218421072439, 'bridge': 0.11300853842290307, 'large-vehicle': 0.295854049973833, 'tennis-court': 0.8918800582306501, 'storage-tank': 0.6269895962862366, 'plane': 0.7730131420976701, 'baseball-diamond': 0.3002366534174908, 'small-vehicle': 0.33292284039195025, 'ground-track-field': 0.4641182195217763, 'soccer-ball-field': 0.544402668494067, 'harbor': 0.20218190583374546},
'0.5': {'roundabout': 0.6352674576203479, 'ship': 0.8009780498350924, 'helicopter': 0.3884850891246363, 'swimming-pool': 0.534946378063368, 'basketball-court': 0.652845483724726, 'mAP': 0.6569046285197613, 'bridge': 0.3962628965531039, 'large-vehicle': 0.5999397914964152, 'tennis-court': 0.8988757684667548, 'storage-tank': 0.8172860055540161, 'plane': 0.8928955787374921, 'baseball-diamond': 0.6814346324626301, 'small-vehicle': 0.6290465093756571, 'ground-track-field': 0.6175511242729564, 'soccer-ball-field': 0.6761322393512399, 'harbor': 0.6316224231579817},
'0.6': {'roundabout': 0.5781157538363746, 'ship': 0.7199207063081612, 'helicopter': 0.3486534066889177, 'swimming-pool': 0.4398985020076733, 'basketball-court': 0.6368261016329557, 'mAP': 0.6045192532757853, 'bridge': 0.32054319940239395, 'large-vehicle': 0.5162125149783844, 'tennis-court': 0.8988757684667548, 'storage-tank': 0.7954587452095123, 'plane': 0.8890395501724334, 'baseball-diamond': 0.5969377624733084, 'small-vehicle': 0.6035867142500114, 'ground-track-field': 0.589625754248992, 'soccer-ball-field': 0.6445403051158263, 'harbor': 0.48955401434507995},
'0.95': {'roundabout': 0.0, 'ship': 0.000263777673104098, 'helicopter': 0.0, 'swimming-pool': 0.0, 'basketball-court': 0.0, 'mAP': 0.012526813891763558, 'bridge': 0.0, 'large-vehicle': 0.0003099173553719008, 'tennis-court': 0.12162723121627231, 'storage-tank': 0.045454545454545456, 'plane': 0.002920774005111354, 'baseball-diamond': 0.001652892561983471, 'small-vehicle': 0.0004722550177095632, 'ground-track-field': 0.0, 'soccer-ball-field': 0.01515151515151515, 'harbor': 4.9299940840070995e-05},
'0.9': {'roundabout': 0.018181818181818184, 'ship': 0.008348794063079777, 'helicopter': 0.0009469696969696969, 'swimming-pool': 0.001041938004688721, 'basketball-court': 0.09090909090909091, 'mAP': 0.07198973955676663, 'bridge': 0.006993006993006993, 'large-vehicle': 0.0058312707135965095, 'tennis-court': 0.5342162059798452, 'storage-tank': 0.12782250147260946, 'plane': 0.1498485366308333, 'baseball-diamond': 0.02097902097902098, 'small-vehicle': 0.0037962037962037958, 'ground-track-field': 0.0303030303030303, 'soccer-ball-field': 0.07683982683982685, 'harbor': 0.0037878787878787876},
'0.65': {'roundabout': 0.5334065574344371, 'ship': 0.699180938894961, 'helicopter': 0.3486534066889177, 'swimming-pool': 0.35950862347304074, 'basketball-court': 0.6266555406785881, 'mAP': 0.5681346675246143, 'bridge': 0.2538510928563938, 'large-vehicle': 0.4736305525634216, 'tennis-court': 0.8988757684667548, 'storage-tank': 0.7705413509430967, 'plane': 0.8828573447357645, 'baseball-diamond': 0.5219317942523523, 'small-vehicle': 0.5640717794973014, 'ground-track-field': 0.5540673038545378, 'soccer-ball-field': 0.6439298704611816, 'harbor': 0.3908580880684668},
'0.55': {'roundabout': 0.6152366299900335, 'ship': 0.731570491648859, 'helicopter': 0.36944993577421675, 'swimming-pool': 0.5140305604450786, 'basketball-court': 0.652845483724726, 'mAP': 0.6327665777581061, 'bridge': 0.35478135684341233, 'large-vehicle': 0.5635620575410195, 'tennis-court': 0.8988757684667548, 'storage-tank': 0.8102992452721097, 'plane': 0.8920391532063079, 'baseball-diamond': 0.6411355718265235, 'small-vehicle': 0.6213305991975432, 'ground-track-field': 0.616276370725309, 'soccer-ball-field': 0.6750784187817075, 'harbor': 0.5349870229279885},
'0.7': {'roundabout': 0.5136619422636372, 'ship': 0.606183142084986, 'helicopter': 0.3429418407679277, 'swimming-pool': 0.2326831491464119, 'basketball-court': 0.5766348484014912, 'mAP': 0.5022796928155692, 'bridge': 0.1691773154585879, 'large-vehicle': 0.39850330001375195, 'tennis-court': 0.8988757684667548, 'storage-tank': 0.7184087415090521, 'plane': 0.7961972288242679, 'baseball-diamond': 0.40392119019168193, 'small-vehicle': 0.4732099822707241, 'ground-track-field': 0.5297768282026153, 'soccer-ball-field': 0.5791869352544582, 'harbor': 0.2948331793771897}}
"""


