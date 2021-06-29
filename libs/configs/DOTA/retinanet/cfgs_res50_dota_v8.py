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
METHOD = 'R'
ANCHOR_RATIOS = [1, 1 / 3., 3.]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA_1x_20210617'

"""
RetinaNet-R + 90
FLOPs: 512292484;    Trainable params: 34524216

AP50:95: [0.6500133314849993, 0.6221115492957145, 0.5783876659070408, 0.5151213008971589, 0.43878296014912604,
          0.3368465304363474, 0.22462667183919535, 0.11392563249065944, 0.03363650833577883, 0.002065178825137116]
mmAP: 0.35155173296611575
++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--
{'0.95': {'bridge': 0.004132231404958678, 'small-vehicle': 9.587038324185702e-05, 'roundabout': 0.0010822510822510823, 'tennis-court': 0.0213903743315508, 'ground-track-field': 0.0, 'basketball-court': 0.0, 'mAP': 0.002065178825137116, 'ship': 0.00033732501264968796, 'large-vehicle': 0.0002361275088547816, 'plane': 0.0008340283569641369, 'helicopter': 0.0, 'harbor': 9.362419249133976e-05, 'baseball-diamond': 0.0, 'storage-tank': 0.002775850104094379, 'swimming-pool': 0.0, 'soccer-ball-field': 0.0}, '0.6': {'bridge': 0.2180613029626103, 'small-vehicle': 0.48955296733750253, 'roundabout': 0.5743715824779693, 'tennis-court': 0.8988701575269984, 'ground-track-field': 0.499228771859173, 'basketball-court': 0.5809814425549822, 'mAP': 0.5783876659070408, 'ship': 0.7601302692501211, 'large-vehicle': 0.7307283640874089, 'plane': 0.8942021611064075, 'helicopter': 0.3072975896469221, 'harbor': 0.3269043269672149, 'baseball-diamond': 0.6129045619299885, 'storage-tank': 0.734913760456796, 'swimming-pool': 0.4140610594981138, 'soccer-ball-field': 0.6336066709434047}, '0.8': {'bridge': 0.09090909090909091, 'small-vehicle': 0.11325009395748946, 'roundabout': 0.21515151515151515, 'tennis-court': 0.7921941866364417, 'ground-track-field': 0.2096861471861472, 'basketball-court': 0.24332649806334017, 'mAP': 0.22462667183919535, 'ship': 0.26291307136684317, 'large-vehicle': 0.18348068300629328, 'plane': 0.40802398035426435, 'helicopter': 0.0303030303030303, 'harbor': 0.03896103896103896, 'baseball-diamond': 0.14207792207792208, 'storage-tank': 0.3159039855551453, 'swimming-pool': 0.014925373134328358, 'soccer-ball-field': 0.3082934609250399}, '0.75': {'bridge': 0.10214211076280041, 'small-vehicle': 0.19545376634203668, 'roundabout': 0.28496697538050925, 'tennis-court': 0.8550385251444838, 'ground-track-field': 0.31532093174321907, 'basketball-court': 0.4440063904349619, 'mAP': 0.3368465304363474, 'ship': 0.47083599337171966, 'large-vehicle': 0.3676002962555352, 'plane': 0.6242575391640972, 'helicopter': 0.04145169898594556, 'harbor': 0.09856749311294766, 'baseball-diamond': 0.2940265227697746, 'storage-tank': 0.4576325952192638, 'swimming-pool': 0.11452184179456906, 'soccer-ball-field': 0.3868752760633457}, '0.65': {'bridge': 0.1715728715728716, 'small-vehicle': 0.38714087806405156, 'roundabout': 0.48752732344925004, 'tennis-court': 0.8963243210004948, 'ground-track-field': 0.46565355697662375, 'basketball-court': 0.55128536051835, 'mAP': 0.5151213008971589, 'ship': 0.730472420280297, 'large-vehicle': 0.6571471502556688, 'plane': 0.8841534431766825, 'helicopter': 0.24147205105658032, 'harbor': 0.23720190635365604, 'baseball-diamond': 0.49210312524188676, 'storage-tank': 0.6753273086899294, 'swimming-pool': 0.30443298751756787, 'soccer-ball-field': 0.5450048093034745}, '0.5': {'bridge': 0.2812488756837439, 'small-vehicle': 0.6122705620423831, 'roundabout': 0.6293850696709108, 'tennis-court': 0.9013591967565613, 'ground-track-field': 0.5750856173584699, 'basketball-court': 0.6068251479708417, 'mAP': 0.6500133314849993, 'ship': 0.8601726316408385, 'large-vehicle': 0.7834049829446229, 'plane': 0.8961770992768222, 'helicopter': 0.3507698724777362, 'harbor': 0.5232626569936892, 'baseball-diamond': 0.6772717890839638, 'storage-tank': 0.7775776682818869, 'swimming-pool': 0.5425412825833318, 'soccer-ball-field': 0.7328475195091895}, '0.7': {'bridge': 0.12460815047021945, 'small-vehicle': 0.2845134660892606, 'roundabout': 0.3834644013719063, 'tennis-court': 0.8925703337005814, 'ground-track-field': 0.38178255494732427, 'basketball-court': 0.5012620086155817, 'mAP': 0.43878296014912604, 'ship': 0.6196205362182818, 'large-vehicle': 0.5254475720737497, 'plane': 0.7717130392038616, 'helicopter': 0.21916545239320936, 'harbor': 0.148302763837693, 'baseball-diamond': 0.394255080184464, 'storage-tank': 0.5963860132836875, 'swimming-pool': 0.22030141020234578, 'soccer-ball-field': 0.5183516196447231}, '0.55': {'bridge': 0.2446933360819818, 'small-vehicle': 0.5572753205907381, 'roundabout': 0.6127007450546773, 'tennis-court': 0.9013591967565613, 'ground-track-field': 0.5460695154831853, 'basketball-court': 0.598903178921024, 'mAP': 0.6221115492957145, 'ship': 0.8395840401868254, 'large-vehicle': 0.747613676166832, 'plane': 0.895943123347051, 'helicopter': 0.3439557489426765, 'harbor': 0.4312764134413297, 'baseball-diamond': 0.6530921050594436, 'storage-tank': 0.7668861840981078, 'swimming-pool': 0.4949977385816317, 'soccer-ball-field': 0.6973229167236527}, '0.9': {'bridge': 0.004132231404958678, 'small-vehicle': 0.003952569169960474, 'roundabout': 0.012396694214876032, 'tennis-court': 0.2869467055233188, 'ground-track-field': 0.045454545454545456, 'basketball-court': 0.012121212121212121, 'mAP': 0.03363650833577883, 'ship': 0.0303030303030303, 'large-vehicle': 0.00306894905544568, 'plane': 0.024475524475524476, 'helicopter': 0.0, 'harbor': 0.00267379679144385, 'baseball-diamond': 0.012987012987012986, 'storage-tank': 0.0202020202020202, 'swimming-pool': 0.0003787878787878788, 'soccer-ball-field': 0.045454545454545456}, '0.85': {'bridge': 0.025974025974025972, 'small-vehicle': 0.0303030303030303, 'roundabout': 0.11091872433199819, 'tennis-court': 0.6556119429916903, 'ground-track-field': 0.08181818181818182, 'basketball-court': 0.1098124098124098, 'mAP': 0.11392563249065944, 'ship': 0.11035613870665417, 'large-vehicle': 0.028601992732106313, 'plane': 0.16961143393022934, 'helicopter': 0.006493506493506493, 'harbor': 0.012987012987012986, 'baseball-diamond': 0.06296914095079233, 'storage-tank': 0.14633945289682992, 'swimming-pool': 0.0036784025223331584, 'soccer-ball-field': 0.1534090909090909}, 'mmAP': 0.35155173296611575}
"""

