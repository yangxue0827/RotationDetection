# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from alpharotate.utils.pretrain_zoo import PretrainModelZoo
from configs._base_.models.retinanet_r50_fpn import *
from configs._base_.datasets.dota_detection import *
from configs._base_.schedules.schedule_1x import *

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 11725 * 2
DECAY_EPOCH = [16, 22, 40]
MAX_EPOCH = 24
WARM_EPOCH = 1 / 16.
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DIOR-R'
CLASS_NUM = 20
IMG_SHORT_SIDE_LEN = [800, 450, 500, 640, 700, 900, 1000, 1100, 1200]
IMG_MAX_LENGTH = 1200
# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = True

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_SUBNET_CONV = 4
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [1, 1 / 2, 2.]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0
REG_LOSS_MODE = 3  # KLD loss

KL_TAU = 1.0
KL_FUNC = 1   # 0: sqrt  1: log

VERSION = 'RetinaNet_DIOR_R_KL_2x_20211116'

"""
RetinaNet-H + kl (fix bug)
FLOPs: 844273828;    Trainable params: 32553441

cls : airplane|| Recall: 0.8425474914758889 || Precison: 0.3559156378600823|| AP: 0.7785884109308336
F1:0.8084297273138955 P:0.9272097053726169 R:0.71663419386264
cls : airport|| Recall: 0.7282282282282282 || Precison: 0.0244221763432197|| AP: 0.47564552781677816
F1:0.5848690208169043 P:0.6641221374045801 R:0.5225225225225225
cls : baseballfield|| Recall: 0.844496214327315 || Precison: 0.19990349486454814|| AP: 0.7597515593804858
F1:0.799207678644462 P:0.9151014274981217 R:0.7093768200349446
cls : basketballcourt|| Recall: 0.9384902143522833 || Precison: 0.10112980165704243|| AP: 0.897341697882234
F1:0.9172735247281774 P:0.965979381443299 R:0.8732525629077353
cls : bridge|| Recall: 0.4723831595210506 || Precison: 0.04447434452161897|| AP: 0.31692865753005095
F1:0.3913200902617238 P:0.5021173623714459 R:0.32058709926612594
cls : chimney|| Recall: 0.8244422890397672 || Precison: 0.04936694157277268|| AP: 0.7477457251685983
F1:0.8448795785435904 P:0.9758576874205845 R:0.7449078564500485
cls : dam|| Recall: 0.7639405204460966 || Precison: 0.022900763358778626|| AP: 0.3654928040309627
F1:0.456702900124476 P:0.4678362573099415 R:0.44609665427509293
cls : Expressway-Service-area|| Recall: 0.9391705069124424 || Precison: 0.07284294803059546|| AP: 0.8327769146716648
F1:0.849946609988696 P:0.8950050968399592 R:0.8092165898617512
cls : Expressway-toll-station|| Recall: 0.6656976744186046 || Precison: 0.03594694294011459|| AP: 0.5895497214068484
F1:0.6893335856715005 P:0.9375 R:0.5450581395348837
cls : golffield|| Recall: 0.9182608695652174 || Precison: 0.06795366795366796|| AP: 0.7952387152510804
F1:0.8250176745665468 P:0.8617424242424242 R:0.7913043478260869
cls : groundtrackfield|| Recall: 0.956498673740053 || Precison: 0.07447028210317624|| AP: 0.823843739079107
F1:0.7964132251986162 P:0.7907949790794979 R:0.8021220159151193
cls : harbor|| Recall: 0.6628019323671498 || Precison: 0.03204758864475139|| AP: 0.40710416976116254
F1:0.48201197028218085 P:0.5325282430853135 R:0.4402576489533011
cls : overpass|| Recall: 0.6739618406285073 || Precison: 0.05843713507201246|| AP: 0.5043461604271156
F1:0.5718072726047254 P:0.7111853088480802 R:0.4781144781144781
cls : ship|| Recall: 0.7437048826237708 || Precison: 0.3729973202577114|| AP: 0.6746865789242859
F1:0.727993413551255 P:0.8435616130602464 R:0.6402830671289718
cls : stadium|| Recall: 0.8988095238095238 || Precison: 0.07390187201761898|| AP: 0.7238433515088328
F1:0.7307324298184406 P:0.8242990654205608 R:0.65625
cls : storagetank|| Recall: 0.5310988399469201 || Precison: 0.38214186712662085|| AP: 0.5133621536590279
F1:0.6090582292052148 P:0.8570873786407767 R:0.47236847737682464
cls : tenniscourt|| Recall: 0.9181533433201688 || Precison: 0.2805659592176446|| AP: 0.8693424109722246
F1:0.8945835599539258 P:0.9527546937519237 R:0.8431158926869127
cls : trainstation|| Recall: 0.787819253438114 || Precison: 0.0315748031496063|| AP: 0.45540672347687494
F1:0.5134278367061836 P:0.5201612903225806 R:0.5068762278978389
cls : vehicle|| Recall: 0.32222222222222224 || Precison: 0.10905302741570749|| AP: 0.2971980176296661
F1:0.382970085599508 P:0.6782442748091603 R:0.2668168168168168
cls : windmill|| Recall: 0.6894596397598399 || Precison: 0.11640479810778848|| AP: 0.556126984332122
F1:0.6598241562501809 P:0.7662108513453904 R:0.5793862575050034
mAP is : 0.6192160011919979

ms
cls : airplane|| Recall: 0.8844374086702387 || Precison: 0.19678660453018315|| AP: 0.7844502759583319
F1:0.8169189385953431 P:0.9002932551319648 R:0.7476863127131028
cls : airport|| Recall: 0.6336336336336337 || Precison: 0.01565630333160199|| AP: 0.4808555525420033
F1:0.6043817111974242 P:0.6584070796460177 R:0.5585585585585585
cls : baseballfield|| Recall: 0.8977868375072802 || Precison: 0.10193420400066126|| AP: 0.7653030556909667
F1:0.7995338771756815 P:0.9203640500568828 R:0.7067559697146185
cls : basketballcourt|| Recall: 0.9496738117427772 || Precison: 0.05632635011884363|| AP: 0.8980385068637642
F1:0.919887876011159 P:0.9632840387557369 R:0.8802423112767941
cls : bridge|| Recall: 0.5477018153727308 || Precison: 0.0330883210827208|| AP: 0.3438495603588649
F1:0.41763532227399963 P:0.5454545454545454 R:0.3383545770567787
cls : chimney|| Recall: 0.8438409311348206 || Precison: 0.031088082901554404|| AP: 0.7737225089545566
F1:0.8478451486093808 P:0.9821200510855683 R:0.7458777885548011
cls : dam|| Recall: 0.5576208178438662 || Precison: 0.01575547502757208|| AP: 0.33127797698997424
F1:0.46399502893325817 P:0.5021645021645021 R:0.4312267657992565
cls : Expressway-Service-area|| Recall: 0.9327188940092166 || Precison: 0.045785639958376693|| AP: 0.8464323987242302
F1:0.876880660455395 P:0.9288659793814433 R:0.8304147465437788
cls : Expressway-toll-station|| Recall: 0.7761627906976745 || Precison: 0.017338788233002143|| AP: 0.6656227840397055
F1:0.7077929324673791 P:0.8622129436325678 R:0.6002906976744186
cls : golffield|| Recall: 0.8417391304347827 || Precison: 0.052262174711154304|| AP: 0.7602532303974886
F1:0.8258377261342238 P:0.8945233265720081 R:0.7669565217391304
cls : groundtrackfield|| Recall: 0.9824933687002653 || Precison: 0.0350187195098892|| AP: 0.8265598513711371
F1:0.8010013026377454 P:0.7617224880382775 R:0.8445623342175066
cls : harbor|| Recall: 0.5516908212560386 || Precison: 0.03531740304723419|| AP: 0.4068872772545807
F1:0.5015663329872048 P:0.6426774031202819 R:0.41127214170692433
cls : overpass|| Recall: 0.6717171717171717 || Precison: 0.039270365145500474|| AP: 0.526083860350754
F1:0.5839744424571963 P:0.6879756468797564 R:0.5072951739618407
cls : ship|| Recall: 0.8808048655715341 || Precison: 0.3511801565987921|| AP: 0.7960662096596614
F1:0.8470324253209885 P:0.908623327582278 R:0.7932700505883021
cls : stadium|| Recall: 0.9508928571428571 || Precison: 0.04220050191520275|| AP: 0.7467583938671385
F1:0.7357034323037165 P:0.8299065420560747 R:0.6607142857142857
cls : storagetank|| Recall: 0.6772398441847524 || Precison: 0.29729780517137705|| AP: 0.6137006085787667
F1:0.7093927616777206 P:0.8705179282868526 R:0.5986045117931595
cls : tenniscourt|| Recall: 0.9460710881111263 || Precison: 0.13634133417070635|| AP: 0.8817760969909002
F1:0.8858590506238846 P:0.9515246286161063 R:0.8286803758681738
cls : trainstation|| Recall: 0.6994106090373281 || Precison: 0.02077376436949291|| AP: 0.4789791533232309
F1:0.5507197287069961 P:0.6365979381443299 R:0.48526522593320237
cls : vehicle|| Recall: 0.4633258258258258 || Precison: 0.07548219811401524|| AP: 0.3840248272065296
F1:0.4672360727789938 P:0.7354458958232543 R:0.3423798798798799
cls : windmill|| Recall: 0.7621747831887925 || Precison: 0.08252970708274641|| AP: 0.6355714527969252
F1:0.7216227980215358 P:0.7610062893081762 R:0.6861240827218146
mAP is : 0.6473106790959756
"""