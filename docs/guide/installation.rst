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
If docker is not available, we provide detailed steps to install the requirements by ``pip``:
::

    pip install -r requirements.txt
    pip install -v -e .  # or "python setup.py develop"

Or, you can simply install AlphaRotate with the following commands:
::

    pip install alpharotate


.. note::
    For 30xx series graphics cards (cuda11), we recommend this `blog <https://blog.csdn.net/qq_39543404/article/details/112171851>`_ to install tf1.xx