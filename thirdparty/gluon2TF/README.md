# Convert ResNets weights from GluonCV to Tensorflow

[Original repository](https://github.com/yangJirui/gluon2TF) is write by [Jirui Yang](https://github.com/yangJirui)

## Abstract
GluonCV released some new resnet pre-training weights and designed some new resnets (such as resnet_v1_b, resnet_v1_d, refer [this](https://arxiv.org/pdf/1812.01187.pdf) for detail).

This project reproduces the resnet in glouncv by Tensorflow and attempts to convert the pre-training weights in glouncv to the Tensorflow CheckPoints.
At present, we have completed the conversion of resnet50_v1_b, resnet101_v1_b, resnet50_v1_d, resnet101_v1_d,
and the 1000-dimensional Logits error rate is controlled within the range of 1e-5.
(We welcome you to submit PR to support more models.)

We also try to transfer these weights to object detection (using FPN as the baseline, the specific detection code we will post [here](https://github.com/DetectionTeamUCAS/FPN_Tensorflow_DEV).),
and **train on voc07trainVal (excluding voc2012 dataset), test in voc07test**. The results are as follows:

## Comparison

### use_voc2007_metric
| Models | mAP | sheep | horse | bicycle | bottle | cow | sofa | bus | dog | cat | person | train | diningtable | aeroplane | car | pottedplant | tvmonitor | chair | bird | boat  | motorbike |
|------------|:---:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|:--:|
|[Faster-RCNN](https://github.com/DetectionTeamUCAS/Faster-RCNN_Tensorflow) resnet101_v1(original)|74.63|76.35|86.18|79.87|58.73|83.4|74.75|80.03|85.4|86.55|78.24|76.07|70.89|78.52|86.26|47.80|76.34|52.14|78.06|58.90|78.04|
|FPN resnet101_v1(original)|76.14|74.63|85.13|81.67|63.79|82.43|77.83|83.07|86.45|85.82|81.08|81.01|71.22|80.01|86.30|48.05|73.89|56.99|78.33|62.91|82.24|
|FPN resnet101_v1_d|77.98|78.01|87.48|85.34|65.42|84.56|74.42|82.97|87.87|87.34|82.14|84.44|70.32|80.64|88.6|51.9|76.59|59.31|81.19|67.84|83.1|


**FPN_resnet101_v1_d is transfer from GluonCV**

**FPN_resnet101_v1(original) is official resnet in [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/slim/nets)**

## My Development Environment
1、python2.7 (anaconda recommend)

2、cuda9.0

3、[opencv(cv2)](https://pypi.org/project/opencv-python/)

4、mxnet-cu90 (1.3.0)

5、tensorflow == 1.10

6、[glouncv](https://github.com/dmlc/gluon-cv)

## Download MxNet GluonCV PreTrained Weights

```
cd $PATH_ROOT/resnet
(modify the resnet version in the main function of download_mxnet_resnet_weights.py.)
python download_mxnet_resnet_weights.py
```


## Convert MxNet Weights To Tensorflow CheckPoint and caculate Erros

modify the main function in gluon2TF/resnet/test_resnet.py as following, and then run it
```
MODEL_NAME = 'resnet101_v1d' (modify the version as u want)
Mxnet_Weights_PATH = '../mxnet_weights/resnet101_v1d-1b2b825f.params' (remember modify the path)

cal_erro(img_path='../demo_img/person.jpg',
             use_tf_ckpt=False,
             ckpt_path='../tf_ckpts/%s.ckpt' % MODEL_NAME,
             save_ckpt=True)
```

Just run it :
```
cd $PATH_ROOT/resnet
python test_resnet
```

## caculate Erros between the converted tensorflow chenckpoints and Mxnet GluonCV Weights

modify the main function in gluon2TF/resnet/test_resnet.py as following, and then run it
```
MODEL_NAME = 'resnet101_v1d' (modify the version as u want)
Mxnet_Weights_PATH = '../mxnet_weights/resnet101_v1d-1b2b825f.params' (remember modify the path)

cal_erro(img_path='../demo_img/person.jpg',
             use_tf_ckpt=True,
             ckpt_path='../tf_ckpts/%s.ckpt' % MODEL_NAME,
             save_ckpt=False)
```

Just run it :
```
cd $PATH_ROOT/resnet
python test_resnet
```

