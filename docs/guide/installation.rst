=============
Installation
=============
Docker
-----------
We recommend using docker images if `docker <https://www.docker.com/>`_ or other container runtimes e.g. `singularity <https://sylabs.io/singularity/>`_ is available on your devices.

We maintain a prebuilt image at `dockerhub <https://hub.docker.com/u/yangxue2docker>`_:
::

    yangxue2docker/yx-tf-det:tensorflow1.13.1-cuda10-gpu-py3

.. note::
    For 30xx series graphics cards (cuda11), please download image from `tensorflow-release-notes <https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_20-11.html#rel_20-11>`_ according to your development environment, e.g. nvcr.io/nvidia/tensorflow:20.11-tf1-py3

Manual configuration
--------------------------
This repository is developed and tested with ubuntu 16.04, python 3.5 (anaconda recommend), tensorflow-gpu 1.13, cuda 10.0, opencv-python 4.1.1.26, tqdm 4.54.0, Shapely 1.7.1, tfplot 0.2.0 (optional).
If docker is not available, we provide detailed steps to install the requirements by ``apt`` and ``pip``.

.. note::
    For 30xx series graphics cards (cuda11), we recommend this `blog <https://blog.csdn.net/qq_39543404/article/details/112171851>`_ to install tf1.xx