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
REG_WEIGHT = 0.01

VERSION = 'RetinaNet_DOTA_RIDet_2x_20210517'

"""
RIDet-8p

AP50:95: [0.640675014551357, 0.6139224270603888, 0.5946688338137965, 0.5587801936045129, 0.49051758602022755, 0.40975916671426993, 0.3021192962301658, 0.1842079180259587, 0.09339241132519678, 0.017017089925002074]
mmAP: 0.39050599372708766
++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--
{'0.5': {'plane': 0.8967040410755641, 'baseball-diamond': 0.6775019950459811, 'bridge': 0.37182707349249194, 'ground-track-field': 0.5727606736995065, 'small-vehicle': 0.612300223663883, 'large-vehicle': 0.5732312492658187, 'ship': 0.7942451990022708, 'tennis-court': 0.9023186917069271, 'basketball-court': 0.5805009674090916, 'storage-tank': 0.8202473567268557, 'soccer-ball-field': 0.6679897821856713, 'roundabout': 0.6506300523681344, 'harbor': 0.6174079592860839, 'swimming-pool': 0.5537103231646051, 'helicopter': 0.3187496301774699, 'mAP': 0.640675014551357}, '0.55': {'plane': 0.8956507077872073, 'baseball-diamond': 0.6504534022833071, 'bridge': 0.34646026188679546, 'ground-track-field': 0.5727606736995065, 'small-vehicle': 0.6034165954719043, 'large-vehicle': 0.5332882871936172, 'ship': 0.7403435451069277, 'tennis-court': 0.9023186917069271, 'basketball-court': 0.5805009674090916, 'storage-tank': 0.8042524736476739, 'soccer-ball-field': 0.6423066747439885, 'roundabout': 0.6081060197856761, 'harbor': 0.5293592880921497, 'swimming-pool': 0.5005036223156609, 'helicopter': 0.2991151947753982, 'mAP': 0.6139224270603888}, '0.6': {'plane': 0.8936878872168594, 'baseball-diamond': 0.6360772733234573, 'bridge': 0.3059426989267465, 'ground-track-field': 0.5588561753201073, 'small-vehicle': 0.5802037379749286, 'large-vehicle': 0.4972714944859465, 'ship': 0.731036615950836, 'tennis-court': 0.9023186917069271, 'basketball-court': 0.5804925542630461, 'storage-tank': 0.7814131104908765, 'soccer-ball-field': 0.6370990340950643, 'roundabout': 0.5977875872650751, 'harbor': 0.47033348788146245, 'swimming-pool': 0.45153534890375463, 'helicopter': 0.29597680940185683, 'mAP': 0.5946688338137965}, '0.65': {'plane': 0.891076697125763, 'baseball-diamond': 0.5936293326162789, 'bridge': 0.23992421661299898, 'ground-track-field': 0.5209928248993965, 'small-vehicle': 0.5276567477188169, 'large-vehicle': 0.43349267230411787, 'ship': 0.7146466887968902, 'tennis-court': 0.9021880751970003, 'basketball-court': 0.5714479568221528, 'storage-tank': 0.7690558968157571, 'soccer-ball-field': 0.6118673991106597, 'roundabout': 0.5804435789088291, 'harbor': 0.38874688262363355, 'swimming-pool': 0.3563427430822082, 'helicopter': 0.28019119143318927, 'mAP': 0.5587801936045129}, '0.7': {'plane': 0.842432095416197, 'baseball-diamond': 0.49797949870210606, 'bridge': 0.16552315608919382, 'ground-track-field': 0.47535612508403474, 'small-vehicle': 0.42503102790213515, 'large-vehicle': 0.3670219499306651, 'ship': 0.6198625124414986, 'tennis-court': 0.9018714759582537, 'basketball-court': 0.560296447092149, 'storage-tank': 0.6911450005570015, 'soccer-ball-field': 0.6033396100902146, 'roundabout': 0.4754803691516908, 'harbor': 0.28631855820243596, 'swimming-pool': 0.212316715542522, 'helicopter': 0.23378924814331514, 'mAP': 0.49051758602022755}, '0.75': {'plane': 0.780404345757515, 'baseball-diamond': 0.392440925070459, 'bridge': 0.12190082644628099, 'ground-track-field': 0.4118985861253357, 'small-vehicle': 0.2600659624817695, 'large-vehicle': 0.2644678174714391, 'ship': 0.49731604911054483, 'tennis-court': 0.9012440855840003, 'basketball-court': 0.5286878305824398, 'storage-tank': 0.6173325992150672, 'soccer-ball-field': 0.5312934386056917, 'roundabout': 0.35501761307730784, 'harbor': 0.16639479914592803, 'swimming-pool': 0.13820473644003056, 'helicopter': 0.17971788560023855, 'mAP': 0.40975916671426993}, '0.8': {'plane': 0.6545493374505955, 'baseball-diamond': 0.23811196753250738, 'bridge': 0.023255813953488372, 'ground-track-field': 0.3636935097465083, 'small-vehicle': 0.12627283245142568, 'large-vehicle': 0.13487929699155385, 'ship': 0.28839474635597595, 'tennis-court': 0.8266952342605206, 'basketball-court': 0.4553319006598104, 'storage-tank': 0.4395331509274919, 'soccer-ball-field': 0.45091028944966083, 'roundabout': 0.24459973418961836, 'harbor': 0.11503496503496503, 'swimming-pool': 0.04894215576708158, 'helicopter': 0.12158450868128289, 'mAP': 0.3021192962301658}, '0.85': {'plane': 0.4143112676532178, 'baseball-diamond': 0.12382445141065831, 'bridge': 0.009090909090909092, 'ground-track-field': 0.20260013182708583, 'small-vehicle': 0.035429091907269235, 'large-vehicle': 0.03776689046150124, 'ship': 0.09220722937428917, 'tennis-court': 0.7903959988857919, 'basketball-court': 0.22404436053489682, 'storage-tank': 0.26012261599601333, 'soccer-ball-field': 0.3629589235668153, 'roundabout': 0.16046831955922866, 'harbor': 0.015673981191222573, 'swimming-pool': 0.018181818181818184, 'helicopter': 0.016042780748663103, 'mAP': 0.1842079180259587}, '0.9': {'plane': 0.15512631417864037, 'baseball-diamond': 0.09090909090909091, 'bridge': 0.0030303030303030303, 'ground-track-field': 0.09902597402597403, 'small-vehicle': 0.005244755244755245, 'large-vehicle': 0.004288529150407603, 'ship': 0.011994949494949496, 'tennis-court': 0.5668376426381337, 'basketball-court': 0.0923406692048163, 'storage-tank': 0.12418987563496234, 'soccer-ball-field': 0.19190112213368027, 'roundabout': 0.025974025974025972, 'harbor': 0.006493506493506493, 'swimming-pool': 0.018181818181818184, 'helicopter': 0.0053475935828877, 'mAP': 0.09339241132519678}, '0.95': {'plane': 0.007628734901462175, 'baseball-diamond': 0.0101010101010101, 'bridge': 0.0, 'ground-track-field': 0.011363636363636364, 'small-vehicle': 0.0001326172004508985, 'large-vehicle': 0.00023014959723820485, 'ship': 0.0008340283569641369, 'tennis-court': 0.19757294763484134, 'basketball-court': 0.0034965034965034965, 'storage-tank': 0.0032085561497326204, 'soccer-ball-field': 0.0202020202020202, 'roundabout': 0.0004861448711716091, 'harbor': 0.0, 'swimming-pool': 0.0, 'helicopter': 0.0, 'mAP': 0.017017089925002074}, 'mmAP': 0.39050599372708766}
"""
