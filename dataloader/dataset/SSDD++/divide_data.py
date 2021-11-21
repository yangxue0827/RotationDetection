# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')
import shutil
import os
import random
import math


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

root_path = '/mnt/nas/dataset_share'

image_path = root_path + '/SSDD++/JPEGImages'
xml_path = root_path + '/SSDD++/Annotations'

image_list = os.listdir(image_path)

image_name = [n.split('.')[0] for n in image_list]

image_output_train = os.path.join(root_path, 'SSDD++/train/JPEGImages')
mkdir(image_output_train)
image_output_test = os.path.join(root_path, 'SSDD++/test/JPEGImages')
mkdir(image_output_test)

xml_train = os.path.join(root_path, 'SSDD++/train/Annotations')
mkdir(xml_train)
xml_test = os.path.join(root_path, 'SSDD++/test/Annotations')
mkdir(xml_test)

for i in image_name:
    if i[-1] in ['1', '9']:
        shutil.copy(os.path.join(image_path, i + '.jpg'), image_output_test)
        shutil.copy(os.path.join(xml_path, i + '.xml'), xml_test)
    else:
        shutil.copy(os.path.join(image_path, i + '.jpg'), image_output_train)
        shutil.copy(os.path.join(xml_path, i + '.xml'), xml_train)

