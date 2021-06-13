# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
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

VERSION = 'FPN_Res50D_DOTA_1x_20210607'

"""
R2CNN
FLOPs: 1024153266;    Trainable params: 41772682
AP50:95: [0.6843397862341891, 0.6582597829548086, 0.6090363401631053, 0.5493843739298462, 0.45985680032563303,
          0.347352813503141, 0.22660028897289625, 0.12234504512293037, 0.04767097639576659, 0.003496791544809234]
mmAP: 0.3708342999147126
++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--
{'0.5': {'plane': 0.8967983811013256, 'baseball-diamond': 0.7039421667352082, 'bridge': 0.414742468214385, 'ground-track-field': 0.647366783144057, 'small-vehicle': 0.6208072744101735, 'large-vehicle': 0.7477497347588187, 'ship': 0.8553848590487019, 'tennis-court': 0.9018384621532947, 'basketball-court': 0.6629714220643683, 'storage-tank': 0.8729910600489074, 'soccer-ball-field': 0.6717656158454788, 'roundabout': 0.6737574131280509, 'harbor': 0.6285853937915624, 'swimming-pool': 0.566932126725616, 'helicopter': 0.3994636323428893, 'mAP': 0.6843397862341891}, '0.55': {'plane': 0.8964512187310847, 'baseball-diamond': 0.697831808652497, 'bridge': 0.38008170567488847, 'ground-track-field': 0.6387596414655303, 'small-vehicle': 0.604170351433914, 'large-vehicle': 0.7377129102210435, 'ship': 0.8368058791758053, 'tennis-court': 0.9018384621532947, 'basketball-court': 0.6293184248521866, 'storage-tank': 0.866369608141524, 'soccer-ball-field': 0.6492098013629114, 'roundabout': 0.6276924664251768, 'harbor': 0.5301970543467442, 'swimming-pool': 0.5012015726506137, 'helicopter': 0.3762558390349152, 'mAP': 0.6582597829548086}, '0.6': {'plane': 0.8951073981417682, 'baseball-diamond': 0.6217386343433741, 'bridge': 0.302500925126434, 'ground-track-field': 0.5683072203962353, 'small-vehicle': 0.5426918953256837, 'large-vehicle': 0.725465931518852, 'ship': 0.8107188653979326, 'tennis-court': 0.8958594290246878, 'basketball-court': 0.6273252063771132, 'storage-tank': 0.8482216822902265, 'soccer-ball-field': 0.5615201707115856, 'roundabout': 0.5709304235824753, 'harbor': 0.4272685985291697, 'swimming-pool': 0.4475331226464019, 'helicopter': 0.29035559903463837, 'mAP': 0.6090363401631053}, '0.65': {'plane': 0.8918072341810429, 'baseball-diamond': 0.5745512157396658, 'bridge': 0.2151915019384899, 'ground-track-field': 0.5070183275222802, 'small-vehicle': 0.4684383735060212, 'large-vehicle': 0.6961469260871753, 'ship': 0.7295346671657972, 'tennis-court': 0.8865012641068536, 'basketball-court': 0.6111959666196954, 'storage-tank': 0.7868214471551233, 'soccer-ball-field': 0.5303616536551422, 'roundabout': 0.5124824051204419, 'harbor': 0.3212120075262913, 'swimming-pool': 0.34545959094428086, 'helicopter': 0.1640430276793913, 'mAP': 0.5493843739298462}, '0.7': {'plane': 0.7756469350705273, 'baseball-diamond': 0.41858101625991717, 'bridge': 0.12469741597815696, 'ground-track-field': 0.3793761097836333, 'small-vehicle': 0.37289748219207186, 'large-vehicle': 0.5958117448117018, 'ship': 0.6178409523258832, 'tennis-court': 0.8718424892682632, 'basketball-court': 0.5222209684630067, 'storage-tank': 0.7603370850929984, 'soccer-ball-field': 0.4636521728786032, 'roundabout': 0.4728814560405654, 'harbor': 0.22052426533327812, 'swimming-pool': 0.2591176689616465, 'helicopter': 0.04242424242424243, 'mAP': 0.45985680032563303}, '0.75': {'plane': 0.6313124097726547, 'baseball-diamond': 0.23454621878165038, 'bridge': 0.06654170571696345, 'ground-track-field': 0.29123817359111476, 'small-vehicle': 0.2645002564033964, 'large-vehicle': 0.4603487781325434, 'ship': 0.47095939811984533, 'tennis-court': 0.7886384376459915, 'basketball-court': 0.4497121126302678, 'storage-tank': 0.6437809021366816, 'soccer-ball-field': 0.3625713367092678, 'roundabout': 0.3346128523179607, 'harbor': 0.0997907508993777, 'swimming-pool': 0.09875185670238673, 'helicopter': 0.012987012987012986, 'mAP': 0.347352813503141}, '0.8': {'plane': 0.42213244346482764, 'baseball-diamond': 0.09620551592382579, 'bridge': 0.01652892561983471, 'ground-track-field': 0.19579117982331148, 'small-vehicle': 0.09684042686959603, 'large-vehicle': 0.25852428517811993, 'ship': 0.2537603732862217, 'tennis-court': 0.7747923243470338, 'basketball-court': 0.25867912575229646, 'storage-tank': 0.5040887730890691, 'soccer-ball-field': 0.210725677830941, 'roundabout': 0.23913176025852081, 'harbor': 0.03296401070698046, 'swimming-pool': 0.036566785170137124, 'helicopter': 0.002272727272727273, 'mAP': 0.22660028897289625}, '0.85': {'plane': 0.2231077351378621, 'baseball-diamond': 0.0303030303030303, 'bridge': 0.012987012987012986, 'ground-track-field': 0.09090909090909091, 'small-vehicle': 0.02097902097902098, 'large-vehicle': 0.11277389277389277, 'ship': 0.0564005135753546, 'tennis-court': 0.6553255430203013, 'basketball-court': 0.168162639048715, 'storage-tank': 0.29583870921029326, 'soccer-ball-field': 0.04500151011778919, 'roundabout': 0.10218603949947233, 'harbor': 0.018181818181818184, 'swimming-pool': 0.0030191211003019122, 'helicopter': 0.0, 'mAP': 0.12234504512293037}, '0.9': {'plane': 0.09090909090909091, 'baseball-diamond': 0.008264462809917356, 'bridge': 0.0008971291866028707, 'ground-track-field': 0.0008576329331046312, 'small-vehicle': 0.00522466039707419, 'large-vehicle': 0.0303030303030303, 'ship': 0.009917355371900825, 'tennis-court': 0.3988939760674698, 'basketball-court': 0.02727272727272727, 'storage-tank': 0.11927596920167499, 'soccer-ball-field': 0.004914004914004914, 'roundabout': 0.012987012987012986, 'harbor': 0.0053475935828877, 'swimming-pool': 0.0, 'helicopter': 0.0, 'mAP': 0.04767097639576659}, '0.95': {'plane': 0.012987012987012986, 'baseball-diamond': 0.0, 'bridge': 0.0, 'ground-track-field': 0.0, 'small-vehicle': 0.0004058441558441558, 'large-vehicle': 0.0001562683126928937, 'ship': 0.00029231218941829873, 'tennis-court': 0.0303030303030303, 'basketball-court': 0.0, 'storage-tank': 0.008264462809917356, 'soccer-ball-field': 0.0, 'roundabout': 0.0, 'harbor': 4.294241422252759e-05, 'swimming-pool': 0.0, 'helicopter': 0.0, 'mAP': 0.003496791544809234}, 'mmAP': 0.3708342999147126}

"""
