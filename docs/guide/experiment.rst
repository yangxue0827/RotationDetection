Run the Experiment
===================

Download Model
-------------------

Pretrain weights
^^^^^^^^^^^^^^^^^^^^^^
Download a pretrain weight you need from the following three options, and then put it to `pretrained_weights <https://github.com/yangxue0827/RotationDetection/blob/main/dataloader/pretrained_weights>`_.

* MxNet pretrain weights (recommend in this repo, default in `NET_NAME <https://github.com/yangxue0827/RotationDetection/blob/main/libs/configs/_base_/models/retinanet_r50_fpn.py>`_): resnet_v1d, resnet_v1b, refer to `gluon2TF <https://github.com/yangxue0827/RotationDetection/blob/main/thirdparty/gluon2TF/README.md>`_.
   1) `Baidu Drive (5ht9) <https://pan.baidu.com/s/1GpqKg0dOaaWmwshvv1qWGg>`_
   2) `Google Drive <https://drive.google.com/drive/folders/1BM8ffn1WnsRRb5RcuAcyJAHX8NS2M1Gz?usp=sharing>`_

* Tensorflow pretrain weights: `resnet50_v1 <http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz>`_, `resnet101_v1 <http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz>`_, `resnet152_v1 <http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz>`_, `efficientnet <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_, `mobilenet_v2 <https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz>`_, darknet53 (`Baidu Drive (1jg2) <https://pan.baidu.com/s/1p8V9aaivo9LNxa_OjXjUwA>`_, `Google Drive <https://drive.google.com/drive/folders/1zyg1bvdmLxNRIXOflo_YmJjNJdpHX2lJ?usp=sharing>`_).

* Pytorch pretrain weights, refer to `pretrain_zoo.py <https://github.com/yangxue0827/RotationDetection/blob/main/dataloader/pretrained_weights/pretrain_zoo.py>`_ and `Others <https://github.com/yangxue0827/RotationDetection/blob/main/OTHERS.md>`_.
   1) `Baidu Drive (oofm) <https://pan.baidu.com/s/16nHwlkPsszBvzhMv4h2IwA>`_
   2) `Google Drive <https://drive.google.com/drive/folders/14Bx6TK4LVadTtzNFTQj293cKYk_5IurH?usp=sharing>`_

Trained weights
^^^^^^^^^^^^^^^^^^^^^^
Please download trained models by this project, then put them to `trained_weights <https://github.com/yangxue0827/RotationDetection/blob/main/output/trained_weights>`_.


Compile
-------------------
::

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

Train
-------------------
* If you want to train your own dataset, please note:
   1) Select the detector and dataset you want to use, and mark them as ``#DETECTOR`` and ``#DATASET`` (such as ``#DETECTOR=retinanet`` and ``#DATASET=DOTA``)
   2) Modify parameters (such as ``CLASS_NUM``, ``DATASET_NAME``, ``VERSION``, etc.) in ``$PATH_ROOT/libs/configs/#DATASET/#DETECTOR/cfgs_xxx.py``
   3) Copy ``$PATH_ROOT/libs/configs/#DATASET/#DETECTOR/cfgs_xxx.py`` to ``$PATH_ROOT/libs/configs/cfgs.py``
   4) Add category information in ``$PATH_ROOT/libs/label_name_dict/label_dict.py``
   5) Add data_name to ``$PATH_ROOT/dataloader/dataset/read_tfrecord.py``

* **Make tfrecord**
If image is very large (such as DOTA dataset), the image needs to be cropped. Take DOTA dataset as a example:
::

   cd $PATH_ROOT/dataloader/dataset/DOTA
   python data_crop.py


If image does not need to be cropped, just convert the annotation file into xml format, refer to `example.xml <https://github.com/yangxue0827/RotationDetection/blob/main/example.xml>`_.
::

   cd $PATH_ROOT/dataloader/dataset/
   python convert_data_to_tfrecord.py --root_dir='/PATH/TO/DOTA/'
                                      --xml_dir='labeltxt'
                                      --image_dir='images'
                                      --save_name='train'
                                      --img_format='.png'
                                      --dataset='DOTA'

* **Start training**
::

   cd $PATH_ROOT/tools/#DETECTOR
   python train.py

Train and Evaluation
----------------------
* For large-scale image, take DOTA dataset as a example (the output file or visualization is in ``$PATH_ROOT/tools/#DETECTOR/test_dota/VERSION``):
::

   cd $PATH_ROOT/tools/#DETECTOR
   python test_dota.py --test_dir='/PATH/TO/IMAGES/'
                       --gpus=0,1,2,3,4,5,6,7
                       -ms (multi-scale testing, optional)
                       -s (visualization, optional)
                       -cn (use cpu nms, slightly better <1% than gpu nms but slower, optional)

or (recommend in this repo, better than multi-scale testing)
::

   python test_dota_sota.py --test_dir='/PATH/TO/IMAGES/'
                            --gpus=0,1,2,3,4,5,6,7
                            -s (visualization, optional)
                            -cn (use cpu nms, slightly better <1% than gpu nms but slower, optional)

.. note::
   In order to set the breakpoint conveniently, the read and write mode of the file is' a+'. If the model of the same ``#VERSION`` needs to be tested again, the original test results need to be deleted.

* For small-scale image, take HRSC2016 dataset as a example:
::

   cd $PATH_ROOT/tools/#DETECTOR
   python test_hrsc2016.py --test_dir='/PATH/TO/IMAGES/'
                           --gpu=0
                           --image_ext='bmp'
                           --test_annotation_path='/PATH/TO/ANNOTATIONS'
                           -s (visualization, optional)

* Tensorboard
::

   cd $PATH_ROOT/output/summary
   tensorboard --logdir=.

.. image:: ../../images/images.png
.. image:: ../../images/scalars.png
