# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
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

# bbox head
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA_1x_20210726'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta)) + 180, sin^2(theta) + cos^2(theta) = 1
[-90, 90]   sin in [-1, 1]   cos in [0, 1]
FLOPs: 485784881;    Trainable params: 33051321

AP50:95: [0.6219588395799995, 0.5900819156180207, 0.5517010615222904, 0.4838942320308083, 0.4041809009083858, 0.29513475504257797, 0.1950400605938477, 0.10536651197643225, 0.042175716518712755, 0.0022298248960640693]
mmAP: 0.32917638186871395
++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--
{'0.95': {'mAP': 0.0022298248960640693, 'storage-tank': 0.0101010101010101, 'ground-track-field': 0.0018181818181818182, 'tennis-court': 0.019762845849802372, 'baseball-diamond': 0.0, 'plane': 0.0009045680687471732, 'small-vehicle': 0.0, 'ship': 9.620009620009621e-05, 'harbor': 0.00012436264146250465, 'soccer-ball-field': 0.0, 'bridge': 0.0, 'large-vehicle': 0.0006402048655569782, 'basketball-court': 0.0, 'helicopter': 0.0, 'swimming-pool': 0.0, 'roundabout': 0.0}, '0.55': {'mAP': 0.5900819156180207, 'storage-tank': 0.7821167176986156, 'ground-track-field': 0.585923162177455, 'tennis-court': 0.89640544738318, 'baseball-diamond': 0.6409370785429295, 'plane': 0.8947763065481882, 'small-vehicle': 0.5225800235695158, 'ship': 0.6589287858729878, 'harbor': 0.42278808647875477, 'soccer-ball-field': 0.6168358732455926, 'bridge': 0.28673021434045987, 'large-vehicle': 0.443459314075796, 'basketball-court': 0.5860753133353813, 'helicopter': 0.39321579219288605, 'swimming-pool': 0.4966172021220622, 'roundabout': 0.623839416686508}, '0.65': {'mAP': 0.4838942320308083, 'storage-tank': 0.703686486850428, 'ground-track-field': 0.4582121295547346, 'tennis-court': 0.8929762974960739, 'baseball-diamond': 0.516330396851967, 'plane': 0.8820386553458407, 'small-vehicle': 0.4341109792559855, 'ship': 0.545882257415277, 'harbor': 0.23702298020522927, 'soccer-ball-field': 0.48594326905482227, 'bridge': 0.13793997570138036, 'large-vehicle': 0.3159858646685459, 'basketball-court': 0.5493799221023796, 'helicopter': 0.2644069261535571, 'swimming-pool': 0.3177772367426935, 'roundabout': 0.5167201030632094}, '0.8': {'mAP': 0.1950400605938477, 'storage-tank': 0.23437333952706677, 'ground-track-field': 0.26611867664499245, 'tennis-court': 0.763372516747552, 'baseball-diamond': 0.11260330578512397, 'plane': 0.4760154511924117, 'small-vehicle': 0.0705332644598394, 'ship': 0.16903432269060098, 'harbor': 0.015777610818933134, 'soccer-ball-field': 0.2528386156572064, 'bridge': 0.045454545454545456, 'large-vehicle': 0.04220779220779221, 'basketball-court': 0.22395577395577396, 'helicopter': 0.08682983682983683, 'swimming-pool': 0.020527859237536656, 'roundabout': 0.14595799769850404}, '0.7': {'mAP': 0.4041809009083858, 'storage-tank': 0.5713030835234201, 'ground-track-field': 0.4411286793357105, 'tennis-court': 0.8750232359772143, 'baseball-diamond': 0.3587099269790509, 'plane': 0.7775672061947545, 'small-vehicle': 0.31359852405275784, 'ship': 0.4820215706587658, 'harbor': 0.1638949393047754, 'soccer-ball-field': 0.4344227064365303, 'bridge': 0.08468408684059858, 'large-vehicle': 0.22843088785212995, 'basketball-court': 0.5295997338562823, 'helicopter': 0.1604646737537103, 'swimming-pool': 0.2158296016933543, 'roundabout': 0.42603465716673267}, 'mmAP': 0.32917638186871395, '0.85': {'mAP': 0.10536651197643225, 'storage-tank': 0.10199029604826304, 'ground-track-field': 0.12953474135292317, 'tennis-court': 0.5438413412390602, 'baseball-diamond': 0.045454545454545456, 'plane': 0.2016801962722456, 'small-vehicle': 0.045454545454545456, 'ship': 0.10533936357128913, 'harbor': 0.009767092411720512, 'soccer-ball-field': 0.13636363636363635, 'bridge': 0.011363636363636364, 'large-vehicle': 0.0087730451366815, 'basketball-court': 0.11842167759095656, 'helicopter': 0.045454545454545456, 'swimming-pool': 0.00974025974025974, 'roundabout': 0.06731875719217492}, '0.9': {'mAP': 0.042175716518712755, 'storage-tank': 0.028409090909090908, 'ground-track-field': 0.045454545454545456, 'tennis-court': 0.27613068993864825, 'baseball-diamond': 0.0048484848484848485, 'plane': 0.03693058210824708, 'small-vehicle': 0.0017575757575757577, 'ship': 0.09090909090909091, 'harbor': 0.002331002331002331, 'soccer-ball-field': 0.09090909090909091, 'bridge': 0.0, 'large-vehicle': 0.001918925401775006, 'basketball-court': 0.03409090909090909, 'helicopter': 0.0, 'swimming-pool': 0.0007639419404125286, 'roundabout': 0.018181818181818184}, '0.6': {'mAP': 0.5517010615222904, 'storage-tank': 0.7683112603670017, 'ground-track-field': 0.5278721455490211, 'tennis-court': 0.8961420380991326, 'baseball-diamond': 0.6077399640401626, 'plane': 0.8912951789284248, 'small-vehicle': 0.48169553694012035, 'ship': 0.6401545188085672, 'harbor': 0.3246909175564914, 'soccer-ball-field': 0.5525343072133332, 'bridge': 0.22316835013203135, 'large-vehicle': 0.3906635166271324, 'basketball-court': 0.5846588273927832, 'helicopter': 0.37095318043973435, 'swimming-pool': 0.42964586334191024, 'roundabout': 0.5859903173985088}, '0.75': {'mAP': 0.29513475504257797, 'storage-tank': 0.43778091624743337, 'ground-track-field': 0.3278443778443778, 'tennis-court': 0.7880930350773967, 'baseball-diamond': 0.17719101354088848, 'plane': 0.6489820655043292, 'small-vehicle': 0.18688721191247495, 'ship': 0.3141654459468437, 'harbor': 0.06063518693276478, 'soccer-ball-field': 0.36103287827425756, 'bridge': 0.05344859639338167, 'large-vehicle': 0.13618491201796437, 'basketball-court': 0.43017442167072706, 'helicopter': 0.11521531100478467, 'swimming-pool': 0.14012085535264343, 'roundabout': 0.249265097918402}, '0.5': {'mAP': 0.6219588395799995, 'storage-tank': 0.80839288248404, 'ground-track-field': 0.5945511971124983, 'tennis-court': 0.89640544738318, 'baseball-diamond': 0.6745779464169431, 'plane': 0.8960316622598585, 'small-vehicle': 0.5393852534010024, 'ship': 0.7264871486973908, 'harbor': 0.5127510533352412, 'soccer-ball-field': 0.6561298946271076, 'bridge': 0.34196740290156447, 'large-vehicle': 0.5099733378468009, 'basketball-court': 0.5916845976680699, 'helicopter': 0.4040989826472015, 'swimming-pool': 0.5368336801446979, 'roundabout': 0.6401121067743961}}
"""



