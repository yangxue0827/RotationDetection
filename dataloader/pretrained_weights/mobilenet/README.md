Download a pretrain weight you need from the following three options, and then put it to $PATH_ROOT/dataloader/pretrained_weights. 
1. Tensorflow pretrain weights: [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz), [resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz), [resnet152_v1](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz), [efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), [mobilenet_v2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz), darknet53 ([Baidu Drive (1jg2)](https://pan.baidu.com/s/1p8V9aaivo9LNxa_OjXjUwA), [Google Drive](https://drive.google.com/drive/folders/1zyg1bvdmLxNRIXOflo_YmJjNJdpHX2lJ?usp=sharing)).      
2. MxNet pretrain weights **(Recommend in this repo)**: resnet_v1d, resnet_v1b, refer to [gluon2TF](./thirdparty/gluon2TF/README.md).    
* [Baidu Drive (5ht9)](https://pan.baidu.com/s/1GpqKg0dOaaWmwshvv1qWGg)          
* [Google Drive](https://drive.google.com/drive/folders/1BM8ffn1WnsRRb5RcuAcyJAHX8NS2M1Gz?usp=sharing)      
3. Pytorch pretrain weights, refer to [pretrain_zoo.py](./dataloader/pretrained_weights/pretrain_zoo.py).
* [Baidu Drive (oofm)](https://pan.baidu.com/s/16nHwlkPsszBvzhMv4h2IwA)          
* [Google Drive](https://drive.google.com/drive/folders/14Bx6TK4LVadTtzNFTQj293cKYk_5IurH?usp=sharing)      

Path tree of pretrained_weight 
```
├── pretrained_weight
│    ├── darknet
│       ├── checkpoint
│       ├── darknet.ckpt.data-00000-of-00001
│       ├── darknet.ckpt.index
│       ├── darknet.ckpt.meta
│    ├── efficientnet
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
│    ├── resnet50.npy   
```  
