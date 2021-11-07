# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../..')
import shutil
import os
import random
import math


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


divide_rate = 0.7

root_path = '/mnt/nas/dataset_share/FDDB'

image_path = root_path + '/images'
polygon_txt_path = root_path + '/xml'

image_list = os.listdir(image_path)

image_name = [n.split('.')[0] for n in image_list]

random.shuffle(image_name)

train_image = image_name[:int(math.ceil(len(image_name)) * divide_rate)]
test_image = image_name[int(math.ceil(len(image_name)) * divide_rate):]

image_output_train = os.path.join(root_path, 'trainval/images')
mkdir(image_output_train)
image_output_test = os.path.join(root_path, 'test/images')
mkdir(image_output_test)

polygon_txt_output_train = os.path.join(root_path, 'trainval/xml')
mkdir(polygon_txt_output_train)
polygon_txt_output_test = os.path.join(root_path, 'test/xml')
mkdir(polygon_txt_output_test)


count = 0
for i in train_image:
    shutil.copy(os.path.join(image_path, i + '.jpg'), os.path.join(image_output_train, 'P{}.jpg'.format(count)))
    shutil.copy(os.path.join(polygon_txt_path, i + '.xml'), os.path.join(polygon_txt_output_train, 'P{}.xml'.format(count)))
    if count % 10 == 0:
        print("process step {}".format(count))
    count += 1

for i in test_image:
    shutil.copy(os.path.join(image_path, i + '.jpg'), os.path.join(image_output_test, 'P{}.jpg'.format(count)))
    shutil.copy(os.path.join(polygon_txt_path, i + '.xml'), os.path.join(polygon_txt_output_test, 'P{}.xml'.format(count)))
    if count % 10 == 0:
        print("process step {}".format(count))
    count += 1

