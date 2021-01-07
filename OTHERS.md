## Export Pb
**Example:**
```  
cd $PATH_ROOT/tools/retinanet
python exportPb.py
```

## SWA Object Detection
Paper: [SWA Object Detection](https://arxiv.org/pdf/2012.12645.pdf)      
Usage:
1. Step1: Train additional n epochs (MAX_ITERATION = SAVE_WEIGHTS_INTE\*20 -> MAX_ITERATION = SAVE_WEIGHTS_INTE\*(20+**n**) in cfgs.py)
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