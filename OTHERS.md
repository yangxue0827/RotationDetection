## Export Pb
**Example:**
```  
cd $PATH_ROOT/tools/retinanet
python exportPb.py
```

## SWA Object Detection
Paper: [SWA Object Detection](https://arxiv.org/pdf/2012.12645.pdf)      
Usage:
1. Step1: Train additional **n** epochs (MAX_ITERATION = SAVE_WEIGHTS_INTE\*20 -> MAX_ITERATION = SAVE_WEIGHTS_INTE\*(20+**n**) in cfgs.py)
    ```  
    cd $PATH_ROOT/tools/retinanet
    python train_swa.py
    ```
2. Step2: Average n trained weights
    ```  
    cd $PATH_ROOT/tools
    python swa.py
    ```
3. Step: Test
    ```  
    cd $PATH_ROOT/tools/retinanet
    python test_dota.py --test_dir='/PATH/TO/IMAGES/'  
                        --gpus=0,1,2,3,4,5,6,7  
                        -ms (multi-scale testing, optional)
                        -s (visualization, optional)
    ``` 
    
## Pyrotch Pretrain Weights Conversion via [MMdnn](https://github.com/Microsoft/MMdnn)
Take resnet50 as an example (torch 1.5.1, torchvision 0.6.1):
1. Step1: 
    ```  
    pip install mmdnn
    mmdownload -f pytorch
    mmdownload -f pytorch -n resnet50 -o ./
    mmtoir -f pytorch -d resnet50 --inputShape 3,224,224 -n imagenet_resnet50.pth
    mmtocode -f tensorflow --IRModelPath resnet50.pb --IRWeightPath resnet50.npy --dstModelPath tf_resnet50.py
    ```
2. Step2: Migrate the generated network structure script to [resnet_pytorch.py](./libs/models/backbones/resnet_pytorch.py), and make some modifications, including the freezing of bn and the first few blocks, the construction of feature_dict variables, etc.
