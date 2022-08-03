# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *
from alpharotate.utils.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
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

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0

BCD_TAU = 2.0
BCD_FUNC = 0   # 0: sqrt  1: log

VERSION = 'RetinaNet_DOTA_R3Det_BCD_2x_20210809'

"""
r3det + bcd + sqrt tau=2
FLOPs: 1032041778;    Trainable params: 37769656

************************************************
AP50:95: [0.6923075196249245, 0.6720137784768743, 0.6494995732026184, 0.6011432799842295, 0.5342122668007445,
          0.43955311647917517, 0.32545428828207074, 0.1892386731826185, 0.07725998443180257, 0.010159949231072721]
mmAP: 0.4190842429696131
++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--
{'mmAP': 0.4190842429696131, '0.95': {'ground-track-field': 0.0, 'bridge': 0.0, 'basketball-court': 0.01515151515151515, 'tennis-court': 0.05569801541294415, 'small-vehicle': 0.0003246753246753247, 'ship': 0.0012453300124533001, 'storage-tank': 0.0015483300154833, 'mAP': 0.010159949231072721, 'plane': 0.045454545454545456, 'helicopter': 0.0, 'baseball-diamond': 0.0, 'swimming-pool': 0.0, 'roundabout': 0.0, 'harbor': 0.0, 'large-vehicle': 0.00267379679144385, 'soccer-ball-field': 0.0303030303030303}, '0.7': {'ground-track-field': 0.5101587578412019, 'bridge': 0.17714201732382262, 'basketball-court': 0.5756978544022391, 'tennis-court': 0.8973106738915606, 'small-vehicle': 0.4719934131981497, 'ship': 0.7612149861940519, 'storage-tank': 0.7521237989246363, 'mAP': 0.5342122668007445, 'plane': 0.7898315006157571, 'helicopter': 0.37611147823468394, 'baseball-diamond': 0.3433083639225506, 'swimming-pool': 0.2247899522667876, 'roundabout': 0.5046019386895241, 'harbor': 0.3133342255832269, 'large-vehicle': 0.7002365516192035, 'soccer-ball-field': 0.6153284893037709}, '0.85': {'ground-track-field': 0.12923703437537432, 'bridge': 0.045454545454545456, 'basketball-court': 0.2548074011265217, 'tennis-court': 0.7685107677655042, 'small-vehicle': 0.057046584278051746, 'ship': 0.19909690948880643, 'storage-tank': 0.25213215924941734, 'mAP': 0.1892386731826185, 'plane': 0.416154592614769, 'helicopter': 0.01515151515151515, 'baseball-diamond': 0.03331872946330777, 'swimming-pool': 0.0606060606060606, 'roundabout': 0.12958612753462165, 'harbor': 0.0303030303030303, 'large-vehicle': 0.17064054941866116, 'soccer-ball-field': 0.27653409090909087}, '0.5': {'ground-track-field': 0.6023213045011936, 'bridge': 0.4082861441738149, 'basketball-court': 0.6039490619011185, 'tennis-court': 0.8979035545662721, 'small-vehicle': 0.6507854976206073, 'ship': 0.8677274359908085, 'storage-tank': 0.8642332090901705, 'mAP': 0.6923075196249245, 'plane': 0.8945040236664252, 'helicopter': 0.5856329800522796, 'baseball-diamond': 0.6476352904297071, 'swimming-pool': 0.5808720454177877, 'roundabout': 0.6452357376382123, 'harbor': 0.6351229327957475, 'large-vehicle': 0.8028264569436598, 'soccer-ball-field': 0.6975771195860639}, '0.6': {'ground-track-field': 0.5774195546303299, 'bridge': 0.305521488259209, 'basketball-court': 0.5919708913944453, 'tennis-court': 0.8979035545662721, 'small-vehicle': 0.6070594293676642, 'ship': 0.8627832700669671, 'storage-tank': 0.8432835309659212, 'mAP': 0.6494995732026184, 'plane': 0.8928305385775872, 'helicopter': 0.5344758365050424, 'baseball-diamond': 0.6031753798539609, 'swimming-pool': 0.4828239335189559, 'roundabout': 0.6095349202473259, 'harbor': 0.5026741344060056, 'large-vehicle': 0.7675267379347106, 'soccer-ball-field': 0.663510397744879}, '0.75': {'ground-track-field': 0.4214147120618708, 'bridge': 0.09305873379099923, 'basketball-court': 0.5132887466969809, 'tennis-court': 0.8957231126102131, 'small-vehicle': 0.3312055709029182, 'ship': 0.7069968958556913, 'storage-tank': 0.6388655206152443, 'mAP': 0.43955311647917517, 'plane': 0.7595259934914426, 'helicopter': 0.2264265417661003, 'baseball-diamond': 0.1692227664324874, 'swimming-pool': 0.11818450343040507, 'roundabout': 0.37538220194128724, 'harbor': 0.21626080741511, 'large-vehicle': 0.5733784775079657, 'soccer-ball-field': 0.5543621626689119}, '0.65': {'ground-track-field': 0.5333659174090867, 'bridge': 0.2428687299167854, 'basketball-court': 0.5792313003142693, 'tennis-court': 0.8979035545662721, 'small-vehicle': 0.5678471467625122, 'ship': 0.8513921657540073, 'storage-tank': 0.779971730349929, 'mAP': 0.6011432799842295, 'plane': 0.8885412558741795, 'helicopter': 0.4406792035384303, 'baseball-diamond': 0.5168166347677807, 'swimming-pool': 0.36073349994048265, 'roundabout': 0.5509001339423002, 'harbor': 0.41307641185770516, 'large-vehicle': 0.737967764501055, 'soccer-ball-field': 0.6558537502686439}, '0.9': {'ground-track-field': 0.01652892561983471, 'bridge': 0.012987012987012986, 'basketball-court': 0.07177033492822966, 'tennis-court': 0.5109431023446112, 'small-vehicle': 0.0303030303030303, 'ship': 0.015408320493066256, 'storage-tank': 0.11142638873731311, 'mAP': 0.07725998443180257, 'plane': 0.1357737478771197, 'helicopter': 0.01515151515151515, 'baseball-diamond': 0.003952569169960474, 'swimming-pool': 0.004784688995215311, 'roundabout': 0.03896103896103896, 'harbor': 0.022727272727272728, 'large-vehicle': 0.05454545454545454, 'soccer-ball-field': 0.11363636363636365}, '0.8': {'ground-track-field': 0.306347637570144, 'bridge': 0.0606060606060606, 'basketball-court': 0.4492510175988041, 'tennis-court': 0.8880673117418388, 'small-vehicle': 0.16638720363916046, 'ship': 0.46733181759938963, 'storage-tank': 0.44449626787540514, 'mAP': 0.32545428828207074, 'plane': 0.6401329510171785, 'helicopter': 0.11280331510594668, 'baseball-diamond': 0.09297520661157024, 'swimming-pool': 0.07382333978078659, 'roundabout': 0.25173160173160175, 'harbor': 0.14056020762105081, 'large-vehicle': 0.36267917361091095, 'soccer-ball-field': 0.42462121212121207}, '0.55': {'ground-track-field': 0.5898839285256059, 'bridge': 0.360602644991428, 'basketball-court': 0.5995359992356287, 'tennis-court': 0.8979035545662721, 'small-vehicle': 0.6426092346055572, 'ship': 0.8661899034566998, 'storage-tank': 0.8580310796907491, 'mAP': 0.6720137784768743, 'plane': 0.8938334822232262, 'helicopter': 0.5560858121069117, 'baseball-diamond': 0.6409783766207335, 'swimming-pool': 0.5354116563857212, 'roundabout': 0.609930177164322, 'harbor': 0.5541701465486935, 'large-vehicle': 0.788255226600731, 'soccer-ball-field': 0.6867854544308349}}
"""
