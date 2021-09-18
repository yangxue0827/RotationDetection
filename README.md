# AlphaRotate: A Rotation Detection Benchmark using TensorFlow

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Documentation: [https://rotationdetection.readthedocs.io/](https://rotationdetection.readthedocs.io/)


## Abstract
[AlphaRotate](https://yangxue0827.github.io/files/alpharotate_yx.pdf) is maintained by [Xue Yang](https://yangxue0827.github.io/) with Shanghai Jiao Tong University supervised by [Prof. Junchi Yan](http://thinklab.sjtu.edu.cn/).

**Papers and codes related to remote sensing/aerial image detection: [DOTA-DOAI](https://github.com/SJTU-Thinklab-Det/DOTA-DOAI) <img src="https://img.shields.io/github/stars/SJTU-Thinklab-Det/DOTA-DOAI?style=social" />.**      

Techniques:
- [x] Dataset: DOTA, HRSC2016, ICDAR2015, ICDAR2017 MLT, MSRA-TD500, UCAS-AOD, FDDB, OHD-SJTU, SSDD++, Total-Text
- [x] Baackbone: [ResNet](https://arxiv.org/abs/1512.03385), [MobileNetV2](https://arxiv.org/abs/1801.04381), [EfficientNet](https://arxiv.org/abs/1905.11946), [DarkNet53](https://arxiv.org/abs/1506.02640)
- [x] Neck: [FPN](https://arxiv.org/abs/1708.02002), [BiFPN](https://arxiv.org/abs/1911.09070)
- [x] Detectors: 
  - [x] [R<sup>2</sup>CNN (Faster-RCNN-H)](https://arxiv.org/abs/1706.09579): [R2CNN_Faster-RCNN_Tensorflow](https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow?style=social" />, [DOTA-DOAI](https://github.com/SJTU-Thinklab-Det/DOTA-DOAI) <img src="https://img.shields.io/github/stars/SJTU-Thinklab-Det/DOTA-DOAI?style=social" />, [R2CNN_FPN_Tensorflow (Deprecated)](https://github.com/yangxue0827/R2CNN_FPN_Tensorflow) <img src="https://img.shields.io/github/stars/yangxue0827/R2CNN_FPN_Tensorflow?style=social" />
  - [x] [RRPN (Faster-RCNN-R)](https://arxiv.org/pdf/1703.01086): [TF code](https://github.com/DetectionTeamUCAS/RRPN_Faster-RCNN_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/RRPN_Faster-RCNN_Tensorflow?style=social" />
  - [x] [SCRDet **(ICCV19)**](https://arxiv.org/abs/1811.07126): [R<sup>2</sup>CNN++](https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow?style=social" />, [IoU-Smooth L1 Loss](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation?style=social" />
  - [x] [RetinaNet-H, RetinaNet-R](https://arxiv.org/abs/1908.05612): [TF code](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation?style=social" />
  - [x] [RefineRetinaNet (CascadeRetinaNet)](https://arxiv.org/abs/1908.05612): [TF code](https://github.com/yangxue0827/RotationDetection) <img src="https://img.shields.io/github/stars/Thinklab-SJTU/R3Det_Tensorflow?style=social" />
  - [x] [ATSS](https://arxiv.org/abs/1912.02424)
  - [x] [FCOS](https://arxiv.org/abs/1904.01355)
  - [x] [RSDet **(AAAI21)**](https://arxiv.org/abs/1911.08299): [TF code](https://github.com/Mrqianduoduo/RSDet-8P-4R) <img src="https://img.shields.io/github/stars/Mrqianduoduo/RSDet-8P-4R?style=social" />
  - [x] [R<sup>3</sup>Det **(AAAI21)**](https://arxiv.org/abs/1908.05612): [TF code](https://github.com/Thinklab-SJTU/R3Det_Tensorflow) <img src="https://img.shields.io/github/stars/Thinklab-SJTU/R3Det_Tensorflow?style=social" />, [Pytorch code](https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection) <img src="https://img.shields.io/github/stars/SJTU-Thinklab-Det/r3det-on-mmdetection?style=social" />
  - [x] [Circular Smooth Label (CSL, **ECCV20**)](https://arxiv.org/abs/2003.05597): [TF code](https://github.com/Thinklab-SJTU/CSL_RetinaNet_Tensorflow) <img src="https://img.shields.io/github/stars/Thinklab-SJTU/CSL_RetinaNet_Tensorflow?style=social" />
  - [x] [Densely Coded Label (DCL, **CVPR21**)](https://arxiv.org/abs/2011.09670): [TF code](https://github.com/Thinklab-SJTU/DCL_RetinaNet_Tensorflow) <img src="https://img.shields.io/github/stars/Thinklab-SJTU/DCL_RetinaNet_Tensorflow?style=social" />
  - [x] [GWD (**ICML21**)](https://arxiv.org/abs/2101.11952): [TF code](https://github.com/yangxue0827/RotationDetection) <img src="https://img.shields.io/github/stars/yangxue0827/RotationDetection?style=social" />, [Pytorch code (YOLOv5-GWD)](https://github.com/zhanggefan/rotmmdet) <img src="https://img.shields.io/github/stars/zhanggefan/rotmmdet?style=social" />
  - [ ] BCD (comng soon!): [TF code](https://github.com/yangxue0827/RotationDetection) <img src="https://img.shields.io/github/stars/yangxue0827/RotationDetection?style=social" />
  - [x] [KLD](https://arxiv.org/abs/2106.01883): [TF code](https://github.com/yangxue0827/RotationDetection) <img src="https://img.shields.io/github/stars/yangxue0827/RotationDetection?style=social" />, [Pytorch code (YOLOv5-KLD)](https://github.com/zhanggefan/rotmmdet) <img src="https://img.shields.io/github/stars/zhanggefan/rotmmdet?style=social" />
  - [x] [RIDet](https://arxiv.org/abs/2103.11636): [Pytorch code](https://github.com/ming71/RIDet) <img src="https://img.shields.io/github/stars/ming71/RIDet?style=social" />
  - [ ] KF (comng soon!): [TF code](https://github.com/yangxue0827/RotationDetection) <img src="https://img.shields.io/github/stars/yangxue0827/RotationDetection?style=social" />
  - [x] Mixed method: R<sup>3</sup>Det-DCL, R<sup>3</sup>Det-GWD, R<sup>3</sup>Det-BCD, R<sup>3</sup>Det-KLD, FCOS-RSDet, R<sup>2</sup>CNN-BCD, R<sup>2</sup>CNN-KF
- [x] Loss: CE, [Focal Loss](https://arxiv.org/abs/1708.02002), [Smooth L1 Loss](https://arxiv.org/abs/1504.08083), [IoU-Smooth L1 Loss](https://arxiv.org/abs/1811.07126), [Modulated Loss](https://arxiv.org/abs/1911.08299)
- [x] [Others](./OTHERS.md): [SWA](https://arxiv.org/pdf/2012.12645.pdf), exportPb, [MMdnn](https://github.com/Microsoft/MMdnn)

The above-mentioned rotation detectors are all modified based on the following horizontal detectors:
- Faster RCNN: [TF code](https://github.com/DetectionTeamUCAS/Faster-RCNN_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/Faster-RCNN_Tensorflow?style=social" />
- R-FCN: [TF code](https://github.com/DetectionTeamUCAS/R-FCN_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/R-FCN_Tensorflow?style=social" />
- FPN: [TF code1](https://github.com/DetectionTeamUCAS/FPN_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/FPN_Tensorflow?style=social" />, [TF code2 (Deprecated)](https://github.com/yangxue0827/FPN_Tensorflow) <img src="https://img.shields.io/github/stars/yangxue0827/FPN_Tensorflow?style=social" />
- Cascade RCNN: [TF code](https://github.com/DetectionTeamUCAS/Cascade-RCNN_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/Cascade-RCNN_Tensorflow?style=social" />
- Cascade FPN RCNN: [TF code](https://github.com/DetectionTeamUCAS/Cascade_FPN_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/Cascade_FPN_Tensorflow?style=social" /> 
- RetinaNet: [TF code](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/RetinaNet_Tensorflow?style=social" />
- RefineDet: [MxNet code](https://github.com/DetectionTeamUCAS/RefineDet_MxNet) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/RefineDet_MxNet?style=social" />
- FCOS: [TF code](https://github.com/DetectionTeamUCAS/FCOS_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/FCOS_Tensorflow?style=social" />, [MxNet code](https://github.com/DetectionTeamUCAS/FCOS_GluonCV) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/FCOS_GluonCV?style=social" />

![3](./images/demo.gif)

## Projects
![0](./images/projects.png)

## Latest Performance

<!-- More results and trained models are available in the [MODEL_ZOO.md](MODEL_ZOO.md). -->

### DOTA (Task1)
**Baseline**

|   Backbone  |  Neck  |  Training/test dataset  | Data Augmentation | Epoch |      
|:-----------:|:------:|:-----------------------:|:-----------------:|:-----:|
| ResNet50_v1d 600->800 | FPN | trainval/test | × | **13 (AP50) or 17 (AP50:95) is enough for baseline (default is 13)** |

| Method | Baseline |    DOTA1.0  |   DOTA1.5   |   DOTA2.0   | Model | Anchor | Angle Pred. | Reg. Loss| Angle Range | Configs |      
|:------------:|:------------:|:-----------:|:----------:|:-----------:|:----------:|:-----------:|:-----------:|:-----------:|:---------:|:---------:|    
| - | [RetinaNet-R](https://arxiv.org/abs/1908.05612) | 67.25 | 56.50 | 42.04 | [Baidu Drive (bi8b)](https://pan.baidu.com/s/1s9bdXWhx307Egdnmz0hLBw) | **R** | Reg. (∆⍬) | smooth L1 | [-90,0)  | [dota1.0,](./libs/configs/DOTA/retinanet/cfgs_res50_dota_v7.py) [dota1.5,](./libs/configs/DOTA1.5/retinanet/cfgs_res50_dota1.5_v7.py) [dota2.0](./libs/configs/DOTA2.0/retinanet/cfgs_res50_dota2.0_v7.py) |
| - | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 64.17 | 56.10 | 43.06 | [Baidu Drive (bi8b)](https://pan.baidu.com/s/1s9bdXWhx307Egdnmz0hLBw) | H | Reg. (∆⍬) | smooth L1 | **[-90,90)**  | [dota1.0,](./libs/configs/DOTA/retinanet/cfgs_res50_dota_v15.py) [dota1.5,](./libs/configs/DOTA1.5/retinanet/cfgs_res50_dota1.5_v15.py) [dota2.0](./libs/configs/DOTA2.0/retinanet/cfgs_res50_dota2.0_v15.py) |
| - | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 65.78 | 57.17 | 43.92 | [Baidu Drive (bi8b)](https://pan.baidu.com/s/1s9bdXWhx307Egdnmz0hLBw) | H | Reg. **(sin⍬, cos⍬)** | smooth L1 | [-90,90)  | [dota1.0,](./libs/configs/DOTA/retinanet/cfgs_res50_dota_atan_v5.py) [dota1.5,](./libs/configs/DOTA1.5/retinanet/cfgs_res50_dota1.5_atan_v5.py) [dota2.0](./libs/configs/DOTA2.0/retinanet/cfgs_res50_dota2.0_atan_v5.py) |
| - | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 65.73 | 58.87 | 44.16 | [Baidu Drive (bi8b)](https://pan.baidu.com/s/1s9bdXWhx307Egdnmz0hLBw) | H | Reg. (∆⍬) | smooth L1 | **[-90,0)**  | [dota1.0,](./libs/configs/DOTA/retinanet/cfgs_res50_dota_v4.py) [dota1.5,](./libs/configs/DOTA1.5/retinanet/cfgs_res50_dota1.5_v4.py) [dota2.0](./libs/configs/DOTA2.0/retinanet/cfgs_res50_dota2.0_v4.py)|
| [IoU-Smooth L1](https://arxiv.org/abs/1811.07126) | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 66.99 | 59.17 | 46.31 | [Baidu Drive (qcvc)](https://pan.baidu.com/s/1NmiEzQmiYi7H0OJMqoryqQ) | H | Reg. (∆⍬) | **iou-smooth L1** | [-90,0)  | [dota1.0,](./libs/configs/DOTA/retinanet/cfgs_res50_dota_v5.py) [dota1.5,](./libs/configs/DOTA1.5/retinanet/cfgs_res50_dota1.5_v5.py) [dota2.0](./libs/configs/DOTA2.0/retinanet/cfgs_res50_dota2.0_v5.py) |
| [RIDet](https://arxiv.org/abs/2103.11636) | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 66.06 | 58.91 | 45.35 | [Baidu Drive (njjv)](https://pan.baidu.com/s/1w3qech_ZmYESVxoiaTbBSw) | H | Quad. | hungarian loss | -  | [dota1.0,](./libs/configs/DOTA/ridet/cfgs_res50_dota_ridet_8p_v1.py) [dota1.5,](./libs/configs/DOTA1.5/ridet/cfgs_res50_dota1.5_ridet_8p_v1.py) [dota2.0](./libs/configs/DOTA2.0/ridet/cfgs_res50_dota2.0_ridet_8p_v1.py) |
| [RSDet](https://arxiv.org/pdf/1911.08299) | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 67.27 | 61.42 | 46.71 | [Baidu Drive (2a1f)](https://pan.baidu.com/s/1m8pSOZX50_Bsv-WuJKo_DQ) | H | Quad. | modulated loss | -  | [dota1.0,](./libs/configs/DOTA/rsdet/cfgs_res50_dota_rsdet_v2.py) [dota1.5,](./libs/configs/DOTA1.5/rsdet/cfgs_res50_dota1.5_rsdet_8p_v2.py) [dota2.0](./libs/configs/DOTA2.0/rsdet/cfgs_res50_dota2.0_rsdet_8p_v2.py) |
| [CSL](https://arxiv.org/abs/2003.05597) | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 67.38 | 58.55 | 43.34 | [Baidu Drive (sdbb)](https://pan.baidu.com/s/1ULnKq0sh1LtjF46YPP4iKA) | H | **Cls.: Gaussian (r=1, w=10)** | smooth L1 | [-90,90) | [dota1.0,](./libs/configs/DOTA/csl/cfgs_res50_dota_v45.py) [dota1.5,](./libs/configs/DOTA1.5/csl/cfgs_res50_dota1.5_csl_v45.py) [dota2.0](./libs/configs/DOTA2.0/csl/cfgs_res50_dota2.0_csl_v45.py) |
| [DCL](https://arxiv.org/abs/2011.09670) | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 67.39 | 59.38 | 45.46 | [Baidu Drive (m7pq)](https://pan.baidu.com/s/1lrdPZm1hVfupPrTnB8cYjw) | H | **Cls.: BCL (w=180/256)** | smooth L1 | [-90,90)  | [dota1.0,](./libs/configs/DOTA/dcl/cfgs_res50_dota_dcl_v5.py) [dota1.5,](./libs/configs/DOTA1.5/dcl/cfgs_res50_dota1.5_dcl_v5.py) [dota2.0](./libs/configs/DOTA2.0/dcl/cfgs_res50_dota2.0_dcl_v5.py) |
| - | [FCOS](https://arxiv.org/abs/1904.01355) | 67.69 | 61.05 | 48.10 | [Baidu Drive (pic4)](https://pan.baidu.com/s/1Ge0RnMHIp3NIfqGAzZZkkA) | - | Quad | smooth L1 | -  | [dota1.0,](./libs/configs/DOTA/fcos/cfgs_res50_dota_fcos_v1.py) [dota1.5,](./libs/configs/DOTA1.5/fcos/cfgs_res50_dota1.5_fcos_v1.py) [dota2.0](./libs/configs/DOTA2.0/fcos/cfgs_res50_dota2.0_fcos_v1.py) |
| [RSDet](https://arxiv.org/pdf/1911.08299) | [FCOS](https://arxiv.org/abs/1904.01355) | 67.91 | 62.18 | 48.81 | [Baidu Drive (8ww5)](https://pan.baidu.com/s/1gH4WT4_ZNyYtUwnuZM5iUg) | - | Quad | modulated loss | -  | [dota1.0,](./libs/configs/DOTA/fcos/cfgs_res50_dota_fcos_v3.py) [dota1.5](./libs/configs/DOTA1.5/fcos/cfgs_res50_dota1.5_fcos_v2.py) [dota2.0](./libs/configs/DOTA2.0/fcos/cfgs_res50_dota2.0_fcos_v2.py) |
| [GWD](https://arxiv.org/abs/2101.11952) | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 68.93 | 60.03 | 46.65 | [Baidu Drive (7g5a)](https://pan.baidu.com/s/1D1pSOoq35lU3OkkUhErLBw) | H | Reg. (∆⍬) | **gwd** | [-90,0)  | [dota1.0,](./libs/configs/DOTA/gwd/cfgs_res50_dota_v10.py) [dota1.5,](./libs/configs/DOTA1.5/gwd/cfgs_res50_dota1.5_v11.py) [dota2.0](./libs/configs/DOTA2.0/gwd/cfgs_res50_dota2.0_v11.py) |
| [GWD](https://arxiv.org/abs/2101.11952) **[+ SWA](https://arxiv.org/pdf/2012.12645.pdf)**  | [RetinaNet-H](https://arxiv.org/abs/1908.05612) |  69.92 | 60.60 | 47.63 | [Baidu Drive (qcn0)](https://pan.baidu.com/s/1kZ2kCMIwJSODgfayypTMNw) | H | Reg. (∆⍬) | gwd | [-90,0)  | [dota1.0,](./libs/configs/DOTA/gwd/cfgs_res50_dota_v10.py) [dota1.5,](./libs/configs/DOTA1.5/gwd/cfgs_res50_dota1.5_v11.py) [dota2.0](./libs/configs/DOTA2.0/gwd/cfgs_res50_dota2.0_v11.py) |
| [BCD]() | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 71.23 | 60.78 | 47.48 | [Baidu Drive (0puk)](https://pan.baidu.com/s/1Yn6xkNlJy-jtMXOY0VKdAA) | H | Reg. (∆⍬) | **bcd** | [-90,0) | [dota1.0,](./libs/configs/DOTA/bcd/cfgs_res50_dota_bcd_v3.py) [dota1.5,](./libs/configs/DOTA1.5/bcd/cfgs_res50_dota1.5_bcd_v6.py) [dota2.0](./libs/configs/DOTA2.0/bcd/cfgs_res50_dota2.0_bcd_v6.py) |
| [KLD](https://arxiv.org/abs/2106.01883) | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 71.28 | 62.50 | 47.69 | [Baidu Drive (o6rv)](https://pan.baidu.com/s/1nymEcwYEWoW4vcIxCYEsyw) | H | Reg. (∆⍬) | **kld** | [-90,0) | [dota1.0,](./libs/configs/DOTA/kl/cfgs_res50_dota_kl_v5.py) [dota1.5,](./libs/configs/DOTA1.5/kl/cfgs_res50_dota1.5_kl_v6.py) [dota2.0](./libs/configs/DOTA2.0/kl/cfgs_res50_dota2.0_kl_v5.py) |
| [KF]() | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 70.64 | 62.71 | 48.04 | [Baidu Drive (o72o)](https://pan.baidu.com/s/1MD-N2XGZ10r2HxSJ8PikUQ) | H | Reg. (∆⍬) | **kf** | [-90,0) | [dota1.0,](./libs/configs/DOTA/kf/cfgs_res50_dota_kf_v3.py) [dota1.5,](./libs/configs/DOTA1.5/kf/cfgs_res50_dota1.5_kf_v3.py) [dota2.0](./libs/configs/DOTA2.0/kf/cfgs_res50_dota2.0_kf_v3.py) |
| [R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612) | [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 70.66 | 62.91 | 48.43 | [Baidu Drive (n9mv)](https://pan.baidu.com/s/1mCKXkTgyBUzmEJz5sng0UA) | H->R | Reg. (∆⍬) | smooth L1 | [-90,0)  | [dota1.0,](./libs/configs/DOTA/r3det/cfgs_res50_dota_r3det_v1.py) [dota1.5,](./libs/configs/DOTA1.5/r3det/cfgs_res50_dota1.5_r3det_v1.py) [dota2.0](./libs/configs/DOTA2.0/r3det/cfgs_res50_dota2.0_r3det_v1.py) |
| [DCL](https://arxiv.org/abs/2011.09670) | [R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612) | 71.21 | 61.98 | 48.71 | [Baidu Drive (eg2s)](https://pan.baidu.com/s/18M0JJ-PBpauPzoImigweyw) | H->R | **Cls.: BCL (w=180/256)** | iou-smooth L1 | [-90,0)->[-90,90)  | [dota1.0,](./libs/configs/DOTA/r3det_dcl/cfgs_res50_dota_r3det_dcl_v1.py) [dota1.5,](./libs/configs/DOTA1.5/r3det_dcl/cfgs_res50_dota1.5_r3det_dcl_v2.py) [dota2.0](./libs/configs/DOTA2.0/r3det_dcl/cfgs_res50_dota2.0_r3det_dcl_v2.py) |
| [GWD](https://arxiv.org/abs/2101.11952) | [R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612) | 71.56 | 63.22 | 49.25 | [Baidu Drive (jb6e)](https://pan.baidu.com/s/1DmluE1rOs377bZQMT2Y0tg) | H->R | Reg. (∆⍬) | **smooth L1->gwd** | [-90,0)  | [dota1.0,](./libs/configs/DOTA/r3det_gwd/cfgs_res50_dota_r3det_gwd_v6.py) [dota1.5,](./libs/configs/DOTA1.5/r3det_gwd/cfgs_res50_1.5_dota_r3det_gwd_v6.py) [dota2.0](./libs/configs/DOTA2.0/r3det_gwd/cfgs_res50_2.0_dota_r3det_gwd_v6.py) |
| [BCD]() | [R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612) | 72.22 | 63.53 | 49.71 | [Baidu Drive (v60g)](https://pan.baidu.com/s/1zlwpt6ptCnzTsJzNJyQaHQ) | H->R | Reg. (∆⍬) | **bcd** | [-90,0)  | [dota1.0,](./libs/configs/DOTA/r3det_bcd/cfgs_res50_dota_r3det_bcd_v2.py) [dota1.5,](./libs/configs/DOTA1.5/r3det_bcd/cfgs_res50_dota1.5_r3det_bcd_v2.py) [dota2.0](./libs/configs/DOTA2.0/r3det_bcd/cfgs_res50_dota2.0_r3det_bcd_v2.py) |
| [KLD](https://arxiv.org/abs/2106.01883) | [R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612) | 71.73 | 65.18 | 50.90 | [Baidu Drive (tq7f)](https://pan.baidu.com/s/1ORscIxy4ccx_bj4knJk1qw) | H->R | Reg. (∆⍬) | **kld** | [-90,0)  | [dota1.0,](./libs/configs/DOTA/r3det_kl/cfgs_res50_dota_r3det_kl_v2.py) [dota1.5,](./libs/configs/DOTA1.5/r3det_kl/cfgs_res50_dota1.5_r3det_kl_v2.py) [dota2.0](./libs/configs/DOTA2.0/r3det_kl/cfgs_res50_dota2.0_r3det_kl_v2.py) |
| [KF]() | [R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612) | 72.28 | 64.69 | 50.41 | [Baidu Drive (u77v)](https://pan.baidu.com/s/1n5eqqqE0j3dhYgXM-4_k5A) | H->R | Reg. (∆⍬) | **kf** | [-90,0)  | [dota1.0,](./libs/configs/DOTA/r3det_kf/cfgs_res50_dota_r3det_kf_v5.py) [dota1.5,](./libs/configs/DOTA1.5/r3det_kf/cfgs_res50_dota1.5_r3det_kf_v4.py) [dota2.0](./libs/configs/DOTA2.0/r3det_kf/cfgs_res50_dota2.0_r3det_kf_v4.py) |
| - | [R<sup>2</sup>CNN (Faster-RCNN)](https://arxiv.org/abs/1706.09579) | 72.27 | 66.45 | 52.35 | [Baidu Drive (02s5)](https://pan.baidu.com/s/1tdIc2IUouVwrwUSIbNdvWg) | H->R | Reg. (∆⍬) | smooth L1 | [-90,0)  | [dota1.0,](./libs/configs/DOTA/r2cnn/cfgs_res50_dota_v1.py) [dota1.5](./libs/configs/DOTA1.5/r2cnn/cfgs_res50_dota1.5_r2cnn_v1.py) [dota2.0](./libs/configs/DOTA2.0/r2cnn/cfgs_res50_dota2.0_r2cnn_v1.py) |

**SOTA**    

| Method | Backbone  | DOTA1.0 | Model | MS | Data Augmentation | Epoch | Configs |
|:-----------:|:-----------:|:----------:|:----------:|:----------:|:-----------:|:----------:|:-----------:|
| R<sup>2</sup>CNN-BCD | ResNet152_v1d-FPN | 79.54 | [Baidu Drive (h2u1)](https://pan.baidu.com/s/1HFTUmXynTEaTGxmwFgpo2A) | √ | √ | 34 | [dota1.0](./libs/configs/DOTA/r2cnn_bcd/cfgs_res152_dota_r2cnn_bcd_v4.py) |
| RetinaNet-BCD | ResNet152_v1d-FPN | 78.52 | [Baidu Drive (0puk)](https://pan.baidu.com/s/1Yn6xkNlJy-jtMXOY0VKdAA) |  √ | √ | 51 | [dota1.0](./libs/configs/DOTA/bcd/cfgs_res152_dota_bcd_v5.py) |
| R<sup>3</sup>Det-BCD | ResNet50_v1d-FPN | 79.08 | [Baidu Drive (v60g)](https://pan.baidu.com/s/1zlwpt6ptCnzTsJzNJyQaHQ) | √ | √ | 51 | [dota1.0](./libs/configs/DOTA/r3det_bcd/cfgs_res50_dota_r3det_bcd_v6.py) |
| R<sup>3</sup>Det-BCD | ResNet152_v1d-FPN | 79.95 | [Baidu Drive (v60g)](https://pan.baidu.com/s/1zlwpt6ptCnzTsJzNJyQaHQ) | √ | √ | 51 | [dota1.0](./libs/configs/DOTA/r3det_bcd/cfgs_res152_dota_r3det_bcd_v3.py) |

**Note:**    
- Single GPU training: [SAVE_WEIGHTS_INTE](./libs/configs/cfgs.py) = iter_epoch * 1 (DOTA1.0: iter_epoch=27000, DOTA1.5: iter_epoch=32000, DOTA2.0: iter_epoch=40000)
- Multi-GPU training (**better**): [SAVE_WEIGHTS_INTE](./libs/configs/cfgs.py) = iter_epoch * 2

## My Development Environment
**docker images: yangxue2docker/yx-tf-det:tensorflow1.13.1-cuda10-gpu-py3 or yangxue2docker/py3-tf1.15.2-nv-torch1.8.0-cuda11:v1.0**        
1. python3.5 (anaconda recommend)               
2. cuda 10.0                     
3. opencv-python 4.1.1.26         
4. [tfplot 0.2.0](https://github.com/wookayin/tensorflow-plot) (optional)            
5. tensorflow-gpu 1.13
6. tqdm 4.54.0
7. Shapely 1.7.1

**Note: For 30xx series graphics cards, I recommend this [blog](https://blog.csdn.net/qq_39543404/article/details/112171851) to install tf1.xx, or refer to [ngc](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) and [tensorflow-release-notes](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_20-11.html#rel_20-11) to download docker image according to your environment, or just use my docker image (yangxue2docker/py3-tf1.15.2-nv-torch1.8.0-cuda11:v1.0)**

## Download Model
### Pretrain weights
Download a pretrain weight you need from the following three options, and then put it to $PATH_ROOT/dataloader/pretrained_weights. 
1. MxNet pretrain weights **(recommend in this repo, default in [NET_NAME](./libs/configs/_base_/models/retinanet_r50_fpn.py))**: resnet_v1d, resnet_v1b, refer to [gluon2TF](./thirdparty/gluon2TF/README.md).    
* [Baidu Drive (5ht9)](https://pan.baidu.com/s/1GpqKg0dOaaWmwshvv1qWGg)          
* [Google Drive](https://drive.google.com/drive/folders/1BM8ffn1WnsRRb5RcuAcyJAHX8NS2M1Gz?usp=sharing)  
2. Tensorflow pretrain weights: [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz), [resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz), [resnet152_v1](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz), [efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), [mobilenet_v2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz), darknet53 ([Baidu Drive (1jg2)](https://pan.baidu.com/s/1p8V9aaivo9LNxa_OjXjUwA), [Google Drive](https://drive.google.com/drive/folders/1zyg1bvdmLxNRIXOflo_YmJjNJdpHX2lJ?usp=sharing)).      
3. Pytorch pretrain weights, refer to [pretrain_zoo.py](./dataloader/pretrained_weights/pretrain_zoo.py) and [Others](./OTHERS.md).
* [Baidu Drive (oofm)](https://pan.baidu.com/s/16nHwlkPsszBvzhMv4h2IwA)          
* [Google Drive](https://drive.google.com/drive/folders/14Bx6TK4LVadTtzNFTQj293cKYk_5IurH?usp=sharing)      


### Trained weights
1. Please download trained models by this project, then put them to $PATH_ROOT/output/pretained_weights.

## Compile
    ```  
    cd $PATH_ROOT/libs/utils/cython_utils
    rm *.so
    rm *.c
    rm *.cpp
    python setup.py build_ext --inplace (or make)
    
    cd $PATH_ROOT/libs/utils/
    rm *.so
    rm *.c
    rm *.cpp
    python setup.py build_ext --inplace
    ```

## Train 

1. If you want to train your own dataset, please note:  
    ```
    (1) Select the detector and dataset you want to use, and mark them as #DETECTOR and #DATASET (such as #DETECTOR=retinanet and #DATASET=DOTA)
    (2) Modify parameters (such as CLASS_NUM, DATASET_NAME, VERSION, etc.) in $PATH_ROOT/libs/configs/#DATASET/#DETECTOR/cfgs_xxx.py
    (3) Copy $PATH_ROOT/libs/configs/#DATASET/#DETECTOR/cfgs_xxx.py to $PATH_ROOT/libs/configs/cfgs.py
    (4) Add category information in $PATH_ROOT/libs/label_name_dict/label_dict.py     
    (5) Add data_name to $PATH_ROOT/dataloader/dataset/read_tfrecord.py  
    ```     

2. Make tfrecord       
    If image is very large (such as DOTA dataset), the image needs to be cropped. Take DOTA dataset as a example:      
    ```  
    cd $PATH_ROOT/dataloader/dataset/DOTA
    python data_crop.py
    ```  
    If image does not need to be cropped, just convert the annotation file into xml format, refer to [example.xml](./example.xml).
    ```  
    cd $PATH_ROOT/dataloader/dataset/  
    python convert_data_to_tfrecord.py --root_dir='/PATH/TO/DOTA/' 
                                       --xml_dir='labeltxt'
                                       --image_dir='images'
                                       --save_name='train' 
                                       --img_format='.png' 
                                       --dataset='DOTA'
    ```      
    

3. Start training
    ```  
    cd $PATH_ROOT/tools/#DETECTOR
    python train.py
    ```

## Test
1. For large-scale image, take DOTA dataset as a example (the output file or visualization is in $PATH_ROOT/tools/#DETECTOR/test_dota/VERSION): 
    ```  
    cd $PATH_ROOT/tools/#DETECTOR
    python test_dota.py --test_dir='/PATH/TO/IMAGES/'  
                        --gpus=0,1,2,3,4,5,6,7  
                        -ms (multi-scale testing, optional)
                        -s (visualization, optional)
                        -cn (use cpu nms, slightly better <1% than gpu nms but slower, optional)
    
    or (recommend in this repo, better than multi-scale testing)
    
    python test_dota_sota.py --test_dir='/PATH/TO/IMAGES/'  
                             --gpus=0,1,2,3,4,5,6,7  
                             -s (visualization, optional)
                             -cn (use cpu nms, slightly better <1% than gpu nms but slower, optional)
    ``` 

    **Notice: In order to set the breakpoint conveniently, the read and write mode of the file is' a+'. If the model of the same #VERSION needs to be tested again, the original test results need to be deleted.**

2. For small-scale image, take HRSC2016 dataset as a example: 
    ```  
    cd $PATH_ROOT/tools/#DETECTOR
    python test_hrsc2016.py --test_dir='/PATH/TO/IMAGES/'  
                            --gpu=0
                            --image_ext='bmp'
                            --test_annotation_path='/PATH/TO/ANNOTATIONS'
                            -s (visualization, optional)
    ``` 

## Tensorboard
```  
cd $PATH_ROOT/output/summary
tensorboard --logdir=.
``` 

![1](./images/images.png)

![2](./images/scalars.png)

## Citation

If you find our code useful for your research, please consider cite.

```
@article{yang2021alpharotate,
    author  = {Yang, Xue and Yan, Junchi},
    title   = {AlphaRotate: A Rotation Detection Benchmark using TensorFlow},
    year    = {2021},
    url     = {https://github.com/yangxue0827/RotationDetection}
}

```

## Reference
1、https://github.com/endernewton/tf-faster-rcnn   
2、https://github.com/zengarden/light_head_rcnn   
3、https://github.com/tensorflow/models/tree/master/research/object_detection    
4、https://github.com/fizyr/keras-retinanet     


