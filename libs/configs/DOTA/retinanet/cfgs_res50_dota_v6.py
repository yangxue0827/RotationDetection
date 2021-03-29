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
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = 1  # IoU-Smooth L1

VERSION = 'RetinaNet_DOTA_1x_20210125'

"""
RetinaNet-H + IoU-Smooth L1 + Train set
FLOPs: 484911740;    Trainable params: 33002916
{'0.85': {'mAP': 0.14627782041763213, 'baseball-diamond': 0.11090909090909092, 'harbor': 0.09090909090909091, 'plane': 0.27835452345970446, 'small-vehicle': 0.06942999791970043, 'ground-track-field': 0.11363636363636365, 'large-vehicle': 0.02268483256923226, 'soccer-ball-field': 0.19731404958677687, 'storage-tank': 0.2602907061740236, 'ship': 0.04735276609934541, 'roundabout': 0.1387608843659297, 'bridge': 0.01515151515151515, 'basketball-court': 0.15643939393939393, 'swimming-pool': 0.008264462809917356, 'tennis-court': 0.6846696287343972, 'helicopter': 0.0},
'0.55': {'mAP': 0.6180425772794623, 'baseball-diamond': 0.6605370152939137, 'harbor': 0.4316249637254168, 'plane': 0.8921551140609973, 'small-vehicle': 0.5763753197159789, 'ground-track-field': 0.5960033673780386, 'large-vehicle': 0.5140964940637592, 'soccer-ball-field': 0.640981123098384, 'storage-tank': 0.7846530324419969, 'ship': 0.738425717459638, 'roundabout': 0.6418239168041917, 'bridge': 0.35065545624653716, 'basketball-court': 0.6259688322347887, 'swimming-pool': 0.5057527087880236, 'tennis-court': 0.8957746574595289, 'helicopter': 0.4158109404207408},
'0.95': {'mAP': 0.011906148214656699, 'baseball-diamond': 0.002457002457002457, 'harbor': 0.0, 'plane': 0.0020512820512820513, 'small-vehicle': 0.0002158079309414621, 'ground-track-field': 0.0, 'large-vehicle': 7.070968958446272e-05, 'soccer-ball-field': 0.0023923444976076554, 'storage-tank': 0.0303030303030303, 'ship': 0.0012202562538133007, 'roundabout': 0.0, 'bridge': 0.0, 'basketball-court': 0.0106951871657754, 'swimming-pool': 0.0, 'tennis-court': 0.1291866028708134, 'helicopter': 0.0},
'0.9': {'mAP': 0.07580978907249437, 'baseball-diamond': 0.0303030303030303, 'harbor': 0.0009609840476648087, 'plane': 0.073397913561848, 'small-vehicle': 0.045454545454545456, 'ground-track-field': 0.09090909090909091, 'large-vehicle': 0.0037878787878787876, 'soccer-ball-field': 0.12794612794612795, 'storage-tank': 0.11706293706293706, 'ship': 0.01515151515151515, 'roundabout': 0.09090909090909091, 'bridge': 0.00505050505050505, 'basketball-court': 0.08181818181818182, 'swimming-pool': 0.0005136106831022085, 'tennis-court': 0.4538814244018972, 'helicopter': 0.0},
'0.75': {'mAP': 0.3417057488239831, 'baseball-diamond': 0.3028093173943694, 'harbor': 0.11737033988004233, 'plane': 0.6720992024059061, 'small-vehicle': 0.25482224509179446, 'ground-track-field': 0.37560100835582394, 'large-vehicle': 0.19102836159387138, 'soccer-ball-field': 0.41438761938286617, 'storage-tank': 0.5774681299092772, 'ship': 0.3730873154069099, 'roundabout': 0.29727064991144037, 'bridge': 0.10824345146379044, 'basketball-court': 0.48384735173571125, 'swimming-pool': 0.060374951467080815, 'tennis-court': 0.7954446866292606, 'helicopter': 0.10173160173160173},
'0.6': {'mAP': 0.5791064575829437, 'baseball-diamond': 0.6248039229596921, 'harbor': 0.32938729188562516, 'plane': 0.8902808281603279, 'small-vehicle': 0.5351314275067095, 'ground-track-field': 0.5590740041047245, 'large-vehicle': 0.45663867418087245, 'soccer-ball-field': 0.6077495705067665, 'storage-tank': 0.7775719402072222, 'ship': 0.7173144329753193, 'roundabout': 0.5802595038036154, 'bridge': 0.2974250151148842, 'basketball-court': 0.6085307288285026, 'swimming-pool': 0.41054800972002176, 'tennis-court': 0.8957746574595289, 'helicopter': 0.3961068563303408},
'0.65': {'mAP': 0.5265410662157112, 'baseball-diamond': 0.5475621917565324, 'harbor': 0.2759891108003529, 'plane': 0.8813266027280913, 'small-vehicle': 0.4844295515247114, 'ground-track-field': 0.5148099523771921, 'large-vehicle': 0.40185132145515895, 'soccer-ball-field': 0.5663829827626151, 'storage-tank': 0.7513131709990646, 'ship': 0.6190257567463207, 'roundabout': 0.515484895036467, 'bridge': 0.2283485419922782, 'basketball-court': 0.5779998260225663, 'swimming-pool': 0.324631937042415, 'tennis-court': 0.8924878958911624, 'helicopter': 0.31647225610073904},
'0.5': {'mAP': 0.6460501958853879, 'baseball-diamond': 0.6870826445522494, 'harbor': 0.5247515287117566, 'plane': 0.8932411340814576, 'small-vehicle': 0.6028646746484635, 'ground-track-field': 0.6105602406816446, 'large-vehicle': 0.5493312315081982, 'soccer-ball-field': 0.6668038658789921, 'storage-tank': 0.7887524010910201, 'ship': 0.7908880246267763, 'roundabout': 0.6814504580260496, 'bridge': 0.37875375467079525, 'basketball-court': 0.6403238109298491, 'swimming-pool': 0.5209860833260855, 'tennis-court': 0.8957746574595289, 'helicopter': 0.45918842808794963},
'0.8': {'mAP': 0.24088790930914045, 'baseball-diamond': 0.17976851136659613, 'harbor': 0.09090909090909091, 'plane': 0.5213444041546939, 'small-vehicle': 0.13348422859614767, 'ground-track-field': 0.21176838718714558, 'large-vehicle': 0.08032623724400782, 'soccer-ball-field': 0.2848878271426679, 'storage-tank': 0.4493570781563396, 'ship': 0.1825232566379315, 'roundabout': 0.21603068176709325, 'bridge': 0.024242424242424242, 'basketball-court': 0.3436968975593106, 'swimming-pool': 0.018181818181818184, 'tennis-court': 0.7858887055827485, 'helicopter': 0.09090909090909091},
'mmAP': 0.36233985492505527,
'0.7': {'mAP': 0.4370708364491412, 'baseball-diamond': 0.4111854636569372, 'harbor': 0.16829072015255023, 'plane': 0.7888077412632399, 'small-vehicle': 0.3765432316965441, 'ground-track-field': 0.4651014925654532, 'large-vehicle': 0.30643126623977845, 'soccer-ball-field': 0.48851302003521874, 'storage-tank': 0.6786500999556405, 'ship': 0.5523497810066385, 'roundabout': 0.40235772917532747, 'bridge': 0.15887277188424623, 'basketball-court': 0.5457928758140458, 'swimming-pool': 0.1969456784320479, 'tennis-court': 0.8849114571496121, 'helicopter': 0.1313092177098376}}
"""


