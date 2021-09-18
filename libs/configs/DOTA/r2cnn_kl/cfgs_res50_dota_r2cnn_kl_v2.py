# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
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

# loss
FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0

KL_TAU = 1.0
KL_FUNC = 1   # 0: sqrt  1: log

VERSION = 'FPN_Res50D_DOTA_KL_1x_20210701'

"""
R2CNN + KLD
AP50:95: [0.6877776443010859, 0.6700094888419666, 0.6334711816882972, 0.579545868925666, 0.49140346828495274,
          0.3925016590071377, 0.2676499419601466, 0.16327984354339217, 0.06795996896120991, 0.004295071798394793]
mmAP: 0.39578941373122495
++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--
{'0.55': {'mAP': 0.6700094888419666, 'soccer-ball-field': 0.5919057019677403, 'small-vehicle': 0.6268579105090586, 'roundabout': 0.6330995906441589, 'tennis-court': 0.9054187075195999, 'plane': 0.8986290403486195, 'ship': 0.8680709636988654, 'helicopter': 0.42111305867286725, 'swimming-pool': 0.53213049200492, 'basketball-court': 0.6583034548841482, 'baseball-diamond': 0.7250585559818273, 'harbor': 0.5481770245402388, 'bridge': 0.37204192024512406, 'storage-tank': 0.8666973511359983, 'large-vehicle': 0.8141041372264812, 'ground-track-field': 0.5885344232498517}, '0.9': {'mAP': 0.06795996896120991, 'soccer-ball-field': 0.024793388429752063, 'small-vehicle': 0.0033766233766233766, 'roundabout': 0.09090909090909091, 'tennis-court': 0.46680290106546785, 'plane': 0.16745767770903158, 'ship': 0.045454545454545456, 'helicopter': 0.0, 'swimming-pool': 0.0006611570247933885, 'basketball-court': 0.03787878787878788, 'baseball-diamond': 0.00879765395894428, 'harbor': 0.0032467532467532465, 'bridge': 0.00039872408293460925, 'storage-tank': 0.11659192825112108, 'large-vehicle': 0.022727272727272728, 'ground-track-field': 0.0303030303030303}, '0.5': {'mAP': 0.6877776443010859, 'soccer-ball-field': 0.5931293571567277, 'small-vehicle': 0.6318721117936019, 'roundabout': 0.6871938680288507, 'tennis-court': 0.9054187075195999, 'plane': 0.8988859017752385, 'ship': 0.8709506101625187, 'helicopter': 0.43651109723477577, 'swimming-pool': 0.5595602334063202, 'basketball-court': 0.6590506938362203, 'baseball-diamond': 0.7303010086914065, 'harbor': 0.6469602750114466, 'bridge': 0.4092352464038929, 'storage-tank': 0.8737405695877584, 'large-vehicle': 0.8211908475987246, 'ground-track-field': 0.5926641363092068}, '0.95': {'mAP': 0.004295071798394793, 'soccer-ball-field': 0.0, 'small-vehicle': 0.00031026993484331366, 'roundabout': 0.0025974025974025974, 'tennis-court': 0.030735049064987963, 'plane': 0.018181818181818184, 'ship': 0.0005254860746190226, 'helicopter': 0.0, 'swimming-pool': 0.0, 'basketball-court': 0.006060606060606061, 'baseball-diamond': 0.0, 'harbor': 0.0, 'bridge': 0.0, 'storage-tank': 0.00546448087431694, 'large-vehicle': 0.0005509641873278236, 'ground-track-field': 0.0}, '0.75': {'mAP': 0.3925016590071377, 'soccer-ball-field': 0.36503200583887513, 'small-vehicle': 0.3016001565191655, 'roundabout': 0.39090770263052077, 'tennis-court': 0.9047259462145943, 'plane': 0.7744111574847867, 'ship': 0.5812542133805486, 'helicopter': 0.06639050604567846, 'swimming-pool': 0.09300940942696005, 'basketball-court': 0.5117316589629708, 'baseball-diamond': 0.16358913229274435, 'harbor': 0.17050455155377015, 'bridge': 0.06480648064806481, 'storage-tank': 0.6447890214843444, 'large-vehicle': 0.48712718083040585, 'ground-track-field': 0.36764576179363506}, '0.65': {'mAP': 0.579545868925666, 'soccer-ball-field': 0.506072825024438, 'small-vehicle': 0.5504278994230408, 'roundabout': 0.532310401685221, 'tennis-court': 0.9052573474932558, 'plane': 0.8928657924970149, 'ship': 0.8407794841646014, 'helicopter': 0.2721949767404313, 'swimming-pool': 0.3224408675099848, 'basketball-court': 0.5959930219288283, 'baseball-diamond': 0.6038743355361746, 'harbor': 0.39276882047018835, 'bridge': 0.23281335738962858, 'storage-tank': 0.7909723095789157, 'large-vehicle': 0.7395328157102101, 'ground-track-field': 0.5148837787330564}, 'mmAP': 0.39578941373122495, '0.85': {'mAP': 0.16327984354339217, 'soccer-ball-field': 0.15, 'small-vehicle': 0.04081129747056495, 'roundabout': 0.1571167462246908, 'tennis-court': 0.7459571429094631, 'plane': 0.4197082397971806, 'ship': 0.10951005176357288, 'helicopter': 0.0036363636363636364, 'swimming-pool': 0.006493506493506493, 'basketball-court': 0.20182367887138364, 'baseball-diamond': 0.012477718360071303, 'harbor': 0.0303030303030303, 'bridge': 0.01515151515151515, 'storage-tank': 0.30118250896567755, 'large-vehicle': 0.09255003115357228, 'ground-track-field': 0.16247582205029013}, '0.7': {'mAP': 0.49140346828495274, 'soccer-ball-field': 0.45922559393477447, 'small-vehicle': 0.45086026717187244, 'roundabout': 0.49627225793605645, 'tennis-court': 0.9052573474932558, 'plane': 0.7964534495403858, 'ship': 0.7347428230307881, 'helicopter': 0.12876334192123667, 'swimming-pool': 0.19729920238126403, 'basketball-court': 0.5449631333709157, 'baseball-diamond': 0.32901304025676265, 'harbor': 0.28521132566972923, 'bridge': 0.12864087930670048, 'storage-tank': 0.7643584617433226, 'large-vehicle': 0.6643946911153169, 'ground-track-field': 0.4855962094019099}, '0.6': {'mAP': 0.6334711816882972, 'soccer-ball-field': 0.5279889097578381, 'small-vehicle': 0.6074722805673629, 'roundabout': 0.6120330087485172, 'tennis-court': 0.9052573474932558, 'plane': 0.8973999969948909, 'ship': 0.8625753247548336, 'helicopter': 0.3792575454912038, 'swimming-pool': 0.4472385205286126, 'basketball-court': 0.6129319841390704, 'baseball-diamond': 0.7109247505640173, 'harbor': 0.5080191352910413, 'bridge': 0.3070606166573874, 'storage-tank': 0.8001248598499632, 'large-vehicle': 0.7536770211406422, 'ground-track-field': 0.5701064233458216}, '0.8': {'mAP': 0.2676499419601466, 'soccer-ball-field': 0.27772508994565626, 'small-vehicle': 0.13745704704275635, 'roundabout': 0.2620846685781751, 'tennis-court': 0.7893020096481993, 'plane': 0.6480408624095461, 'ship': 0.2835207811885127, 'helicopter': 0.025174825174825177, 'swimming-pool': 0.03285870755750274, 'basketball-court': 0.3746591892361796, 'baseball-diamond': 0.0547453510499783, 'harbor': 0.0844265439202148, 'bridge': 0.0303030303030303, 'storage-tank': 0.5150694724299125, 'large-vehicle': 0.26355221251515815, 'ground-track-field': 0.2358293384025506}}
"""
