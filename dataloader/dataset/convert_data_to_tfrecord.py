# -*- coding: utf-8 -*-
# Author: Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>, <yangxue0827@126.com>
#         Jirui Yang <yangjirui123@gmail.com>
# License: Apache-2.0 license
# Copyright (c) SJTU. ALL rights reserved.

from __future__ import division, print_function, absolute_import

import glob
import os
import sys
import xml.etree.cElementTree as ET

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append('../../')

from alpharotate.libs.label_name_dict.label_dict import LabelMap
from alpharotate.utils.tools import makedirs
from configs import cfgs
from alpharotate.utils.order_points import re_order

tf.app.flags.DEFINE_string('root_dir', '/data/dataset/DOTA/crop/trainval/', 'root dir')
tf.app.flags.DEFINE_string('xml_dir', 'labeltxt', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'images', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', '../tfrecord/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.png', 'format of image')
tf.app.flags.DEFINE_string('dataset', 'DOTA', 'dataset')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """
    label_map = LabelMap(cfgs)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = label_map.name2label()[child_item.text]
                if child_item.tag in ['bndbox', 'polygon', 'robndbox']:
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(float(node.text))
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)  # int32 is important

    return img_height, img_width, gtbox_label


def convert_pascal_to_tfrecord():

    # assert FLAGS.dataset == cfgs.DATASET_NAME, 'Keep the FLAGS.dataset == cfgs.DATASET_NAME.'

    xml_path = os.path.join(FLAGS.root_dir, FLAGS.xml_dir)
    image_path = os.path.join(FLAGS.root_dir, FLAGS.image_dir)
    save_path = os.path.join(FLAGS.save_dir, FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord')
    makedirs(FLAGS.save_dir)

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=save_path)
    all_xml = glob.glob(xml_path + '/*.xml')
    total_data = len(all_xml)
    assert total_data != 0, 'Your dataset is empty, please check the data path.'
    pbar = tqdm(total=total_data)
    for count, xml in enumerate(all_xml):

        img_name = xml.split('/')[-1].split('.')[0] + FLAGS.img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)

        # For quad. detection in this repo, such as RSdet, FCOS
        gtbox_label = np.array(re_order(gtbox_label, True), np.int32)

        # if img_height != 600 or img_width != 600:
        #     continue

        img = cv2.imread(img_path)[:, :, ::-1]

        feature = tf.train.Features(feature={
            # do not need encode() in linux
            'img_name': _bytes_feature(img_name.encode()),
            # 'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        pbar.set_description("Image: %s" % img_name)

        pbar.update(1)

        # view_bar('Conversion progress', count + 1, len(glob.glob(xml_path + '/*.xml')))

    # print('\nConversion is complete!')
    writer.close()


if __name__ == '__main__':
    # xml_path = '../data/dataset/VOCdevkit/VOC2007/Annotations/000005.xml'
    # read_xml_gtbox_and_label(xml_path)

    convert_pascal_to_tfrecord()
