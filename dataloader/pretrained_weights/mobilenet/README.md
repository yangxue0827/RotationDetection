1. Please download [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz), [resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz), [resnet152_v1](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz), [efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), [mobilenet_v2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz), darknet53 ([Baidu Drive (1jg2)](https://pan.baidu.com/s/1p8V9aaivo9LNxa_OjXjUwA), [Google Drive](https://drive.google.com/drive/folders/1zyg1bvdmLxNRIXOflo_YmJjNJdpHX2lJ?usp=sharing)) pre-trained models on Imagenet, put them to $PATH_ROOT/dataloader/pretrained_weights.       
2. **(Recommend in this repo)** Or you can choose to use better backbones (resnet_v1d), refer to [gluon2TF](https://github.com/yangJirui/gluon2TF).    
* [Baidu Drive (5ht9)](https://pan.baidu.com/s/1GpqKg0dOaaWmwshvv1qWGg)          
* [Google Drive](https://drive.google.com/drive/folders/1BM8ffn1WnsRRb5RcuAcyJAHX8NS2M1Gz?usp=sharing)      
3. Path tree of pretrained_weight 
```
├── pretrained_weight
│   ├── darknet
│       ├── checkpoint
│       ├── darknet.ckpt.data-00000-of-00001
│       ├── darknet.ckpt.index
│       ├── darknetREADME.md.ckpt.meta
│   ├── efficientnet
│       ├── efficientnet-b0
│           ├── checkpoint
│           ├── model.ckpt.data-00000-of-00001
│           ├── model.ckpt.index
│           ├── model.ckpt.meta
│    ├── mobilenet
│       ├── mobilenet_v1_0.25_128.ckpt.data-00000-of-00001
│       ├── mobilenet_v1_0.25_128.ckpt.index
│       ├── mobilenet_v1_0.25_128.ckpt.meta
│       ├── mobilenet_v1_0.25_128.tflite
│       ├── mobilenet_v1_0.25_128_eval.pbtxt
│       ├── mobilenet_v1_0.25_128_frozen.pb
│       ├── mobilenet_v1_0.25_128_info.txt
│    ├── resnet_v1_50.ckpt    
│    ├── resnet50_v1d.ckpt.index    
│    ├── resnet50_v1d.ckpt.data-00000-of-00001    
│    ├── resnet50_v1d.ckpt.meta    
```  
