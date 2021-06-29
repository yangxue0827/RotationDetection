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
REG_LOSS_MODE = None

VERSION = 'FCOS_DOTA_1x_20210616'

"""
FCOS
FLOPs: 468484100;    Trainable params: 32090136
AP50:95: [0.657299425287604, 0.6241752209258004, 0.5771224126904312, 0.5226585606820188, 0.44830279891608976,
          0.3569762828353407, 0.2545520110329005, 0.1552287494963553, 0.057532759028607015, 0.008582559012697001]
mmAP: 0.3662430779907845
++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--
{'0.5': {'plane': 0.8851570639970467, 'baseball-diamond': 0.6306127210807415, 'bridge': 0.4177832468389531, 'ground-track-field': 0.4931685224392381, 'small-vehicle': 0.6027799633246981, 'large-vehicle': 0.6661781662930775, 'ship': 0.7641297116105128, 'tennis-court': 0.8953644232693438, 'basketball-court': 0.6835566610500415, 'storage-tank': 0.8615740936231205, 'soccer-ball-field': 0.6519606425916767, 'roundabout': 0.669075001875759, 'harbor': 0.6050080530943827, 'swimming-pool': 0.5740066155848368, 'helicopter': 0.45913649264063144, 'mAP': 0.657299425287604}, '0.55': {'plane': 0.8773401921080234, 'baseball-diamond': 0.5882832150156297, 'bridge': 0.36601623649065135, 'ground-track-field': 0.4644889853998973, 'small-vehicle': 0.5821353796788801, 'large-vehicle': 0.6108299657875689, 'ship': 0.7533977349215739, 'tennis-court': 0.8935957870108021, 'basketball-court': 0.6167167532317788, 'storage-tank': 0.8499283753040496, 'soccer-ball-field': 0.6450199555282644, 'roundabout': 0.631729621797236, 'harbor': 0.5124942122847805, 'swimming-pool': 0.530493954219431, 'helicopter': 0.4401579451084402, 'mAP': 0.6241752209258004}, '0.6': {'plane': 0.8343469168872588, 'baseball-diamond': 0.47998571996205536, 'bridge': 0.2949871445251151, 'ground-track-field': 0.429248807073207, 'small-vehicle': 0.5451808201705569, 'large-vehicle': 0.5923391096465394, 'ship': 0.7323895108556205, 'tennis-court': 0.8918719339754393, 'basketball-court': 0.6167167532317788, 'storage-tank': 0.7896539464914193, 'soccer-ball-field': 0.6256437044190803, 'roundabout': 0.5822160910481705, 'harbor': 0.3911822309127578, 'swimming-pool': 0.4623056776438838, 'helicopter': 0.38876782351358624, 'mAP': 0.5771224126904312}, '0.65': {'plane': 0.7674602391012482, 'baseball-diamond': 0.39138267523398085, 'bridge': 0.22779722692766172, 'ground-track-field': 0.38227138540598427, 'small-vehicle': 0.48124259553054344, 'large-vehicle': 0.5606505203129073, 'ship': 0.6449503011199688, 'tennis-court': 0.8882547554383928, 'basketball-court': 0.5996495579090106, 'storage-tank': 0.7740037478302471, 'soccer-ball-field': 0.5890564227707867, 'roundabout': 0.5266423982671122, 'harbor': 0.27965898269520834, 'swimming-pool': 0.3512809815252128, 'helicopter': 0.37557662016201626, 'mAP': 0.5226585606820188}, '0.7': {'plane': 0.7480741942299921, 'baseball-diamond': 0.28491417176356304, 'bridge': 0.15422371408276112, 'ground-track-field': 0.34193771346922236, 'small-vehicle': 0.3800016165394783, 'large-vehicle': 0.4873424696443632, 'ship': 0.5971974256616549, 'tennis-court': 0.8844407748906453, 'basketball-court': 0.57019059163727, 'storage-tank': 0.6811209806639307, 'soccer-ball-field': 0.5139624028187078, 'roundabout': 0.468581334238386, 'harbor': 0.1879641389757574, 'swimming-pool': 0.17323546761840766, 'helicopter': 0.25135498750720736, 'mAP': 0.44830279891608976}, '0.75': {'plane': 0.638480491268426, 'baseball-diamond': 0.16595623005889915, 'bridge': 0.10635310635310635, 'ground-track-field': 0.28253134684456893, 'small-vehicle': 0.2374290083207463, 'large-vehicle': 0.37961045921485476, 'ship': 0.41281075154430213, 'tennis-court': 0.8595147355887445, 'basketball-court': 0.4950299319632033, 'storage-tank': 0.5722552204278989, 'soccer-ball-field': 0.43429171291541124, 'roundabout': 0.3682496734973011, 'harbor': 0.11899620074744276, 'swimming-pool': 0.11274833508133607, 'helicopter': 0.1703870387038704, 'mAP': 0.3569762828353407}, '0.8': {'plane': 0.5051838168885886, 'baseball-diamond': 0.11419068736141907, 'bridge': 0.045454545454545456, 'ground-track-field': 0.18824315994591587, 'small-vehicle': 0.12037899418595661, 'large-vehicle': 0.21871403506702516, 'ship': 0.25618136638116157, 'tennis-court': 0.7677670667094157, 'basketball-court': 0.3549626701800615, 'storage-tank': 0.44570292801193634, 'soccer-ball-field': 0.35910600950204913, 'roundabout': 0.24770569260824132, 'harbor': 0.061633281972265024, 'swimming-pool': 0.07244985061886469, 'helicopter': 0.0606060606060606, 'mAP': 0.2545520110329005}, '0.85': {'plane': 0.286499373204074, 'baseball-diamond': 0.09090909090909091, 'bridge': 0.01674641148325359, 'ground-track-field': 0.11586452762923352, 'small-vehicle': 0.024932135636471557, 'large-vehicle': 0.1137692716640085, 'ship': 0.08762125759106315, 'tennis-court': 0.6380042107163609, 'basketball-court': 0.19159825681564813, 'storage-tank': 0.25665402937253856, 'soccer-ball-field': 0.2582726616817526, 'roundabout': 0.16942148760330578, 'harbor': 0.012987012987012986, 'swimming-pool': 0.004545454545454546, 'helicopter': 0.0606060606060606, 'mAP': 0.1552287494963553}, '0.9': {'plane': 0.04070705426127113, 'baseball-diamond': 0.022727272727272728, 'bridge': 0.00505050505050505, 'ground-track-field': 0.004914004914004914, 'small-vehicle': 0.005509641873278237, 'large-vehicle': 0.011363636363636364, 'ship': 0.019762845849802372, 'tennis-court': 0.38972621438374866, 'basketball-court': 0.08629776021080368, 'storage-tank': 0.09508026695526696, 'soccer-ball-field': 0.15641511990245227, 'roundabout': 0.02097902097902098, 'harbor': 0.0021853146853146855, 'swimming-pool': 0.002272727272727273, 'helicopter': 0.0, 'mAP': 0.057532759028607015}, '0.95': {'plane': 0.0009921865310678407, 'baseball-diamond': 0.0, 'bridge': 0.0, 'ground-track-field': 0.0, 'small-vehicle': 0.0008045052292839903, 'large-vehicle': 0.0001607114158675738, 'ship': 0.004784688995215311, 'tennis-court': 0.036045879795879796, 'basketball-court': 0.018181818181818184, 'storage-tank': 0.045454545454545456, 'soccer-ball-field': 0.004132231404958678, 'roundabout': 0.018181818181818184, 'harbor': 0.0, 'swimming-pool': 0.0, 'helicopter': 0.0, 'mAP': 0.008582559012697001}, 'mmAP': 0.3662430779907845}
"""



