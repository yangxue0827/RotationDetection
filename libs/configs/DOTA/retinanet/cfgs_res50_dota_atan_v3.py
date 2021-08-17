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
FLOPs: 485782956;    Trainable params: 33051321

AP50:95: [0.6322754110454248, 0.6059311773043418, 0.5632382934096964, 0.49771862859839333, 0.4081989007351669,
          0.3062602848457231, 0.18251355775266292, 0.09169461921353883, 0.030156325021323906, 0.001406378024366967]
mmAP: 0.3319393575950639
++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--
{'0.7': {'baseball-diamond': 0.3563174595960667, 'small-vehicle': 0.3470556058828665, 'soccer-ball-field': 0.4388168507702429, 'plane': 0.7721724667790091, 'harbor': 0.16784285197328988, 'basketball-court': 0.527799573057816, 'tennis-court': 0.8876455013326554, 'mAP': 0.4081989007351669, 'ground-track-field': 0.3947477365650395, 'large-vehicle': 0.24898907335659134, 'roundabout': 0.39411262481903564, 'swimming-pool': 0.19184187924012697, 'helicopter': 0.24409613375130615, 'storage-tank': 0.5646867334675699, 'ship': 0.4656468992237672, 'bridge': 0.12121212121212122}, '0.85': {'baseball-diamond': 0.022727272727272728, 'small-vehicle': 0.009041913500512226, 'soccer-ball-field': 0.13279857397504455, 'plane': 0.22933610229969198, 'harbor': 0.012987012987012986, 'basketball-court': 0.0689568496020109, 'tennis-court': 0.5507378093319512, 'mAP': 0.09169461921353883, 'ground-track-field': 0.09956709956709955, 'large-vehicle': 0.008119968211188279, 'roundabout': 0.04247196373180625, 'swimming-pool': 0.0066518847006651885, 'helicopter': 0.0303030303030303, 'storage-tank': 0.0781416862224943, 'ship': 0.060850848316029375, 'bridge': 0.022727272727272728}, '0.65': {'baseball-diamond': 0.4984457419949718, 'small-vehicle': 0.44267883828244364, 'soccer-ball-field': 0.5117791930981501, 'plane': 0.8771255307557164, 'harbor': 0.2776062562106536, 'basketball-court': 0.552904343012926, 'tennis-court': 0.8912619145014375, 'mAP': 0.49771862859839333, 'ground-track-field': 0.49046221611283625, 'large-vehicle': 0.322766359985716, 'roundabout': 0.5062755749094723, 'swimming-pool': 0.33892116045850046, 'helicopter': 0.32635376319586845, 'storage-tank': 0.705944477895066, 'ship': 0.5340001115830795, 'bridge': 0.1892539469790597}, 'mmAP': 0.3319393575950639, '0.5': {'baseball-diamond': 0.6535621959467298, 'small-vehicle': 0.5456807806323374, 'soccer-ball-field': 0.6860827048994753, 'plane': 0.8947737597792101, 'harbor': 0.5363751478860779, 'basketball-court': 0.6176873154724334, 'tennis-court': 0.8918129393565426, 'mAP': 0.6322754110454248, 'ground-track-field': 0.5791792836885137, 'large-vehicle': 0.5181247680724166, 'roundabout': 0.6493765607541576, 'swimming-pool': 0.5339754350861612, 'helicopter': 0.47216033728402657, 'storage-tank': 0.8185746834654506, 'ship': 0.7020284030686756, 'bridge': 0.3847368502891633}, '0.95': {'baseball-diamond': 0.0, 'small-vehicle': 7.783312577833126e-05, 'soccer-ball-field': 0.0, 'plane': 0.0007652280379553106, 'harbor': 0.0, 'basketball-court': 0.0, 'tennis-court': 0.012626262626262626, 'mAP': 0.001406378024366967, 'ground-track-field': 0.0, 'large-vehicle': 0.00015487068297971194, 'roundabout': 0.0, 'swimming-pool': 0.0, 'helicopter': 0.0, 'storage-tank': 0.006993006993006993, 'ship': 0.0004784688995215311, 'bridge': 0.0}, '0.55': {'baseball-diamond': 0.6320131630287713, 'small-vehicle': 0.5307182325496789, 'soccer-ball-field': 0.6620764941453733, 'plane': 0.8932393035821292, 'harbor': 0.4878141166110275, 'basketball-court': 0.6154399403667143, 'tennis-court': 0.8918129393565426, 'mAP': 0.6059311773043418, 'ground-track-field': 0.5649377769597885, 'large-vehicle': 0.45528543765810237, 'roundabout': 0.6310440785417407, 'swimming-pool': 0.48251443479891754, 'helicopter': 0.4694866427195251, 'storage-tank': 0.7857189409259591, 'ship': 0.6486130338194324, 'bridge': 0.3382531245014232}, '0.75': {'baseball-diamond': 0.22251950947603122, 'small-vehicle': 0.1843409687224323, 'soccer-ball-field': 0.3621203220647057, 'plane': 0.643465815892077, 'harbor': 0.11667280088332721, 'basketball-court': 0.47277588721438984, 'tennis-court': 0.7980984934924288, 'mAP': 0.3062602848457231, 'ground-track-field': 0.2743432493432494, 'large-vehicle': 0.14004280543691316, 'roundabout': 0.2686931057337247, 'swimming-pool': 0.09281643448439535, 'helicopter': 0.20213330972824645, 'storage-tank': 0.4226666212549223, 'ship': 0.2859793274376682, 'bridge': 0.10723562152133581}, '0.8': {'baseball-diamond': 0.054945054945054944, 'small-vehicle': 0.05909854751772014, 'soccer-ball-field': 0.2423279433694761, 'plane': 0.41685373386808916, 'harbor': 0.016645326504481434, 'basketball-court': 0.2073187583461683, 'tennis-court': 0.77149325685176, 'mAP': 0.18251355775266292, 'ground-track-field': 0.21605881151335696, 'large-vehicle': 0.05928289604420803, 'roundabout': 0.1201301880494047, 'swimming-pool': 0.026365848263658485, 'helicopter': 0.11056511056511056, 'storage-tank': 0.2339648740615349, 'ship': 0.1647742285111321, 'bridge': 0.03787878787878788}, '0.9': {'baseball-diamond': 0.002840909090909091, 'small-vehicle': 0.00202020202020202, 'soccer-ball-field': 0.0606060606060606, 'plane': 0.030960672048737155, 'harbor': 0.012987012987012986, 'basketball-court': 0.01515151515151515, 'tennis-court': 0.2965832735577859, 'mAP': 0.030156325021323906, 'ground-track-field': 0.00202020202020202, 'large-vehicle': 0.002005347593582888, 'roundabout': 0.002797202797202797, 'swimming-pool': 0.0011755485893416929, 'helicopter': 0.0, 'storage-tank': 0.018867924528301886, 'ship': 0.004329004329004329, 'bridge': 0.0}, '0.6': {'baseball-diamond': 0.5947605740193932, 'small-vehicle': 0.5055344538711746, 'soccer-ball-field': 0.5943352652627037, 'plane': 0.8904069693551632, 'harbor': 0.3802615440126324, 'basketball-court': 0.5710847612841813, 'tennis-court': 0.8915088605364763, 'mAP': 0.5632382934096964, 'ground-track-field': 0.537560462228919, 'large-vehicle': 0.40686780910630754, 'roundabout': 0.5832688892084376, 'swimming-pool': 0.4108415483525353, 'helicopter': 0.40275095765088714, 'storage-tank': 0.7685680786757434, 'ship': 0.6294386018228827, 'bridge': 0.28138562575800985}}
"""



