# UranusDet

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Abstract
This is a tensorflow-based rotation detection benchmark, also called UranusDet.     
UranusDet is written and maintained by [Xue Yang](https://yangxue0827.github.io/) with Shanghai Jiao Tong University supervised by [Prof. Junchi Yan](http://thinklab.sjtu.edu.cn/).

**Papers and codes related to remote sensing/aerial image detection: [DOTA-DOAI](https://github.com/SJTU-Thinklab-Det/DOTA-DOAI).**      

Techniques:
- [x] Dataset: DOTA, HRSC2016, ICDAR2015, ICDAR2017 MLT, MSRA-TD500, UCAS-AOD, FDDB, OHD-SJTU, SSDD++
- [x] Baackbone: [ResNet](https://arxiv.org/abs/1512.03385), [MobileNetV2](https://arxiv.org/abs/1801.04381), [EfficientNet](https://arxiv.org/abs/1905.11946), [DarkNet53](https://arxiv.org/abs/1506.02640)
- [x] Neck: [FPN](https://arxiv.org/abs/1708.02002), [BiFPN](https://arxiv.org/abs/1911.09070)
- [x] Detectors: 
  - [x] [R<sup>2</sup>CNN (Faster-RCNN-H)](https://arxiv.org/abs/1706.09579): [TF code](https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow?style=social" />
  - [ ] [RRPN (Faster-RCNN-R)](https://arxiv.org/pdf/1703.01086): [TF code](https://github.com/DetectionTeamUCAS/RRPN_Faster-RCNN_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/RRPN_Faster-RCNN_Tensorflow?style=social" />
  - [x] [SCRDet **(ICCV19)**](https://arxiv.org/abs/1811.07126): [R<sup>2</sup>CNN++](https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow?style=social" />, [IoU-Smooth L1 Loss](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation)
  - [x] [RetinaNet-H, RetinaNet-R](https://arxiv.org/abs/1908.05612): [TF code](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation?style=social" />
  - [x] [RefineRetinaNet (CascadeRetinaNet)](https://arxiv.org/abs/1908.05612): [TF code](https://github.com/yangxue0827/RotationDetection) <img src="https://img.shields.io/github/stars/Thinklab-SJTU/R3Det_Tensorflow?style=social" />
  - [ ] [FCOS](https://arxiv.org/abs/1904.01355): [TF code](https://github.com/DetectionTeamUCAS/FCOS_Tensorflow) <img src="https://img.shields.io/github/stars/DetectionTeamUCAS/FCOS_Tensorflow?style=social" />
  - [x] [RSDet **(AAAI21)**](https://arxiv.org/abs/1911.08299): [TF code](https://github.com/Mrqianduoduo/RSDet-8P-4R) <img src="https://img.shields.io/github/stars/Mrqianduoduo/RSDet-8P-4R?style=social" />
  - [x] [R<sup>3</sup>Det **(AAAI21)**](https://arxiv.org/abs/1908.05612): [TF code](https://github.com/Thinklab-SJTU/R3Det_Tensorflow) <img src="https://img.shields.io/github/stars/Thinklab-SJTU/R3Det_Tensorflow?style=social" />, [Pytorch code](https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection) <img src="https://img.shields.io/github/stars/SJTU-Thinklab-Det/r3det-on-mmdetection?style=social" />
  - [x] [Circular Smooth Label (CSL, **ECCV20**)](https://arxiv.org/abs/2003.05597): [TF code](https://github.com/Thinklab-SJTU/CSL_RetinaNet_Tensorflow) <img src="https://img.shields.io/github/stars/Thinklab-SJTU/CSL_RetinaNet_Tensorflow?style=social" />
  - [x] [Densely Coded Label (DCL, **CVPR21**)](https://arxiv.org/abs/2011.09670): [TF code](https://github.com/Thinklab-SJTU/DCL_RetinaNet_Tensorflow) <img src="https://img.shields.io/github/stars/Thinklab-SJTU/DCL_RetinaNet_Tensorflow?style=social" />
  - [ ] [GWD](https://arxiv.org/abs/2101.11952): coming soon! <img src="https://img.shields.io/github/stars/yangxue0827/RotationDetection?style=social" />
  - [x] Mixed method: R<sup>3</sup>Det-DCL, R<sup>3</sup>Det-GWD
- [x] Loss: CE, [Focal Loss](https://arxiv.org/abs/1708.02002), [Smooth L1 Loss](https://arxiv.org/abs/1504.08083), [IoU-Smooth L1 Loss](https://arxiv.org/abs/1811.07126), [Modulated Loss](https://arxiv.org/abs/1911.08299)
- [x] [Others](./OTHERS.md): [SWA](https://arxiv.org/pdf/2012.12645.pdf), exportPb, [MMdnn](https://github.com/Microsoft/MMdnn)

![3](demo.gif)

## Projects
![0](projects.png)

## Latest Performance

More results and trained models are available in the [MODEL_ZOO.md](MODEL_ZOO.md).

### DOTA (Task1)
Base setting: 

|   Backbone  |  Neck  |  Training/test dataset  | Data Augmentation | Epoch |      
|:-----------:|:------:|:-----------------------:|:-----------------:|:-----:|
| ResNet50_v1d 600->800 | FPN | trainval/test | × | **13 (single gpu) or 17 (multi-gpu) is enough for baseline (default is 20)** |

| Model |    DOTA1.0   | Model |   DOTA1.5   | Model |   DOTA2.0   | Model | Anchor | Angle Pred. | Reg. Loss| Angle Range | Configs |      
|:------------:|:-----------:|:----------:|:-----------:|:----------:|:-----------:|:----------:|:-----------:|:-----------:|:-----------:|:---------:|:---------:|    
| [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 64.17 | [Baidu Drive (j5l0)](https://pan.baidu.com/s/1Qh_LE6QeGsOBYqMzjAESsA) | - | - | - | - | H | Reg. (∆⍬) | smooth L1 | **180**  | [dota1.0](./libs/configs/DOTA/retinanet/cfgs_res50_dota_v15.py) |
| [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 65.17 | [Baidu Drive (b3f5)](https://pan.baidu.com/s/12uQhzVrrGqHr-x2cNTojXw) | - | - | - | - | H | Reg. **(sin⍬, cos⍬)** | smooth L1 | 180  | [dota1.0](./libs/configs/DOTA/retinanet/cfgs_res50_dota_atan_v1.py) |
| [RetinaNet-H](https://arxiv.org/abs/1908.05612) | 65.73 | [Baidu Drive (jum2)](https://pan.baidu.com/s/19-hEtCGxLfYuluTATQJpdg) | 58.87 | [Baidu Drive (lld0)](https://pan.baidu.com/s/15pqhPVJ6XzvIMLjVZh6aDw) | 44.16 | [Baidu Drive (ffmo)](https://pan.baidu.com/s/1LgDJV2mS6dDhhVwujz_K8A) | H | Reg. (∆⍬) | smooth L1 | **90**  | [dota1.0,](./libs/configs/DOTA/retinanet/cfgs_res50_dota_v4.py) [dota1.5,](./libs/configs/DOTA1.5/retinanet/cfgs_res50_dota1.5_v4.py) [dota2.0](./libs/configs/DOTA2.0/retinanet/cfgs_res50_dota2.0_v4.py)|
| [IoU-Smooth L1](https://arxiv.org/abs/1811.07126) | 66.99 | [Baidu Drive (bc83)](https://pan.baidu.com/s/19lyx6WvThr61xrbpkC9nQg) |||  |  | H | Reg. (∆⍬) | **iou-smooth L1** | 90  | [dota1.0](./libs/configs/DOTA/retinanet/cfgs_res50_dota_v5.py) |
| [RSDet](https://arxiv.org/pdf/1911.08299) | 67.27 | [Baidu Drive (6nt5)](https://pan.baidu.com/s/1-4iXqRMvCOIEtrMFwtXyew) |||  |  | H | Reg. (∆⍬) | modulated loss | -  | [dota1.0](./libs/configs/DOTA/rsdet/cfgs_res50_dota_rsdet_v2.py) |
| [CSL](https://arxiv.org/abs/2003.05597) | 67.38 | [Baidu Drive (g3wt)](https://pan.baidu.com/s/1nrIs-oYA53qQzlPjqYkMJQ) |||  |  | H | **Cls.: Gaussian (r=1, w=10)** | smooth L1 | 180 | [dota1.0](./libs/configs/DOTA/csl/cfgs_res50_dota_v45.py) |
| [DCL](https://arxiv.org/abs/2011.09670) | 67.39 | [Baidu Drive (p9tu)](https://pan.baidu.com/s/1TZ9V0lTTQnMhiepxK1mdqg) |||  |  | H | **Cls.: BCL (w=180/256)** | smooth L1 | 180  | [dota1.0](./libs/configs/DOTA/dcl/cfgs_res50_dota_dcl_v5.py) |
| [GWD](https://arxiv.org/abs/2101.11952) | 68.93 | [Baidu Drive (nb7w)](https://pan.baidu.com/s/1u74Uk0wQQT_8QWR9rYwgrw) | 60.03 | [Baidu Drive (4p2r)](https://pan.baidu.com/s/1t8TqnMOASalTeM1zq9w4HA) | 46.65 | [Baidu Drive (9kwq)](https://pan.baidu.com/s/1k3PhaaZMypw9vpunD9SU9Q) | H | Reg. (∆⍬) | **gwd** | 90  | [dota1.0,](./libs/configs/DOTA/gwd/cfgs_res50_dota_v10.py) [dota1.5,](./libs/configs/DOTA1.5/gwd/cfgs_res50_dota1.5_v11.py) [dota2.0](./libs/configs/DOTA2.0/gwd/cfgs_res50_dota2.0_v11.py) |
| [GWD](https://arxiv.org/abs/2101.11952) **[+ SWA](https://arxiv.org/pdf/2012.12645.pdf)**   | 69.92 | [Baidu Drive (nb7w)](https://pan.baidu.com/s/1u74Uk0wQQT_8QWR9rYwgrw) | - | - | - | - | H | Reg. (∆⍬) | gwd | 90  | [dota1.0](./libs/configs/DOTA/gwd/cfgs_res50_dota_v10.py) |
| new work | 71.28 | [Baidu Drive ()]() | 62.50 | [Baidu Drive ()]() | 47.69 | [Baidu Drive ()]() | H | Reg. (∆⍬) |  | 90 |  |
| [R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612) | 70.66 | [Baidu Drive (30lt)](https://pan.baidu.com/s/143sGeLNjXzcpxi9GV7FVyA) | 62.91 | [Baidu Drive (rdc6)](https://pan.baidu.com/s/1H9VLJyEHR5Y2Yvnvb2TGDg) |  |  | H->R | Reg. (∆⍬) | smooth L1 | 90  | [dota1.0,](./libs/configs/DOTA/r3det/cfgs_res50_dota_r3det_v1.py) [dota1.5](./libs/configs/DOTA1.5/r3det/cfgs_res50_dota1.5_r3det_v1.py) |
| **[R<sup>3</sup>Det-DCL](https://arxiv.org/abs/2011.09670)** | 71.21 | [Baidu Drive (jueq)](https://pan.baidu.com/s/1XR31i3T-C5R16giBxQUNWw) |||  |  | H->R | **Cls.: BCL (w=180/256)** | iou-smooth L1 | 90->180  | [dota1.0](./libs/configs/DOTA/r3det_dcl/cfgs_res50_dota_r3det_dcl_v1.py) |
| **[R<sup>3</sup>Det-GWD](https://arxiv.org/abs/2101.11952)** | 71.56 | [Baidu Drive (8962)](https://pan.baidu.com/s/17_nhbq35YU7WLBvad3TasQ) |||  |  | H->R | Reg. (∆⍬) | **smooth L1->gwd** | 90  | [dota1.0](./libs/configs/DOTA/r3det_gwd/cfgs_res50_dota_r3det_gwd_v6.py) |
| [R<sup>2</sup>CNN (Faster-RCNN)](https://arxiv.org/abs/1706.09579) | 72.27 | [Baidu Drive (wt2b)](https://pan.baidu.com/s/1R_31U2jl7gj6OMvirURnsg) |||  |  | H->R | Reg. (∆⍬) | smooth L1 | 90  | [dota1.0](./libs/configs/DOTA/r2cnn/cfgs_res50_dota_v1.py) |

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

**Note: For 30xx series graphics cards, I recommend this [blog](https://blog.csdn.net/qq_39543404/article/details/112171851) to install tf1.xx, [ngc](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow), and [tensorflow-release-notes](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_20-10.html#rel_20-10). Or just use my docker image (yangxue2docker/py3-tf1.15.2-nv-torch1.8.0-cuda11:v1.0)**

## Download Model
### Pretrain weights
Download a pretrain weight you need from the following three options, and then put it to $PATH_ROOT/dataloader/pretrained_weights. 
1. Tensorflow pretrain weights: [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz), [resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz), [resnet152_v1](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz), [efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), [mobilenet_v2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz), darknet53 ([Baidu Drive (1jg2)](https://pan.baidu.com/s/1p8V9aaivo9LNxa_OjXjUwA), [Google Drive](https://drive.google.com/drive/folders/1zyg1bvdmLxNRIXOflo_YmJjNJdpHX2lJ?usp=sharing)).      
2. MxNet pretrain weights **(Recommend in this repo)**: resnet_v1d, resnet_v1b, refer to [gluon2TF](./thirdparty/gluon2TF/README.md).    
* [Baidu Drive (5ht9)](https://pan.baidu.com/s/1GpqKg0dOaaWmwshvv1qWGg)          
* [Google Drive](https://drive.google.com/drive/folders/1BM8ffn1WnsRRb5RcuAcyJAHX8NS2M1Gz?usp=sharing)      
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
    (5) Add data_name to $PATH_ROOT/data/io/read_tfrecord.py  
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
    python convert_data_to_tfrecord.py --VOC_dir='/PATH/TO/DOTA/' 
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

![1](images.png)

![2](scalars.png)

## Citation

If you find our code useful for your research, please consider cite.

```
@article{yang2021rethinking,
    title={Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss},
    author={Yang, Xue and Yan, Junchi and Qi, Ming and Wang, Wentao and Xiaopeng, Zhang and Qi, Tian},
    journal={arXiv preprint arXiv:2101.11952},
    year={2021}
}

@inproceedings{yang2020dense,
    title={Dense Label Encoding for Boundary Discontinuity Free Rotation Detection},
    author={Yang, Xue and Hou, Liping and Zhou, Yue and Wang, Wentao and Yan, Junchi},
    journal={Proceedings of the IEEE Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}

@inproceedings{yang2020arbitrary,
    title={Arbitrary-oriented object detection with circular smooth label},
    author={Yang, Xue and Yan, Junchi},
    booktitle={European Conference on Computer Vision (ECCV)},
    pages={677--694},
    year={2020},
    organization={Springer}
}

@inproceedings{yang2021r3det,
    title={R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object},
    author={Yang, Xue and Yan, Junchi and Feng, Ziming and He, Tao},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
    year={2021}
}

@inproceedings{qian2021learning,
    title={Learning modulated loss for rotated object detection},
    author={Qian, Wen and Yang, Xue and Peng, Silong and Yan, Junchi and Guo, Yue },
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
    year={2021}
}

@article{yang2020scrdet++,
    title={SCRDet++: Detecting Small, Cluttered and Rotated Objects via Instance-Level Feature Denoising and Rotation Loss Smoothing},
    author={Yang, Xue and Yan, Junchi and Yang, Xiaokang and Tang, Jin and Liao, Wenglong and He, Tao},
    journal={arXiv preprint arXiv:2004.13316},
    year={2020}
}

@inproceedings{yang2019scrdet,
    title={SCRDet: Towards more robust detection for small, cluttered and rotated objects},
    author={Yang, Xue and Yang, Jirui and Yan, Junchi and Zhang, Yue and Zhang, Tengfei and Guo, Zhi and Sun, Xian and Fu, Kun},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
    pages={8232--8241},
    year={2019}
}

```

## Reference
1、https://github.com/endernewton/tf-faster-rcnn   
2、https://github.com/zengarden/light_head_rcnn   
3、https://github.com/tensorflow/models/tree/master/research/object_detection    
4、https://github.com/fizyr/keras-retinanet     


