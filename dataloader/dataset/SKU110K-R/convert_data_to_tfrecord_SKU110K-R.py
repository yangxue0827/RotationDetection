# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
import numpy as np
import tensorflow as tf
import math
import cv2
import os
import json

sys.path.append('../../../')

from utils.tools import makedirs, view_bar

tf.app.flags.DEFINE_string('root_dir', '/data/dataset/SKU110K/', 'Voc dir')
tf.app.flags.DEFINE_string('json_file', 'SKU110K-R-Json/sku110k-r_train.json', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'SKU110K-R/images', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', '../../tfrecord/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.jpg', 'format of image')
tf.app.flags.DEFINE_string('dataset', 'SKU110K-R', 'dataset')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def coordinate_convert_r(box):
    w, h = box[2:-1]
    theta = -box[-1]
    x_lu, y_lu = -w/2, h/2
    x_ru, y_ru = w/2, h/2
    x_ld, y_ld = -w/2, -h/2
    x_rd, y_rd = w/2, -h/2

    x_lu_ = math.cos(theta)*x_lu + math.sin(theta)*y_lu + box[0]
    y_lu_ = -math.sin(theta)*x_lu + math.cos(theta)*y_lu + box[1]

    x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
    y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

    x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
    y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

    x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
    y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

    convert_box = [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]

    return np.around(convert_box)


def convert_pascal_to_tfrecord():
    json_file = os.path.join(FLAGS.root_dir, FLAGS.json_file)
    image_path = os.path.join(FLAGS.root_dir, FLAGS.image_dir)
    save_path = os.path.join(FLAGS.save_dir, FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord')
    makedirs(FLAGS.save_dir)

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=save_path)

    with open(json_file, 'r') as fr:
        all_gts = json.load(fr)
        images = all_gts['images']
        annotations = all_gts['annotations']

    all_gt_label = {}

    for annotation in annotations:
        image_id = annotation['image_id']
        # print(image_id-1)  # 57533
        if image_id > len(images):
            continue
        if images[image_id - 1]['file_name'] in all_gt_label.keys():
            # all_gt_label[images[image_id - 1]['file_name']]['gtboxes'].append(annotation['segmentation'])
            all_gt_label[images[image_id - 1]['file_name']]['gtboxes'].append(coordinate_convert_r(annotation['rbbox']))
            all_gt_label[images[image_id - 1]['file_name']]['labels'].append(annotation['category_id'])
        else:
            all_gt_label[images[image_id - 1]['file_name']] = {'height': images[image_id - 1]['height'],
                                                               'width': images[image_id - 1]['width'],
                                                               # 'gtboxes': [annotation['segmentation']],
                                                               'gtboxes': [coordinate_convert_r(annotation['rbbox'])],
                                                               'labels': [annotation['category_id']]}
    count = 0
    for img_name in all_gt_label.keys():
        img = cv2.imread(os.path.join(image_path, img_name))
        img_height = all_gt_label[img_name]['height']
        img_width = all_gt_label[img_name]['width']
        gtboxes = np.array(all_gt_label[img_name]['gtboxes']).reshape([-1, 8])
        labels = np.array(all_gt_label[img_name]['labels']).reshape([-1, 1])
        gtboxes_and_label = np.array(np.concatenate([gtboxes, labels], axis=-1), np.int32)

        feature = tf.train.Features(feature={
            # do not need encode() in linux
            'img_name': _bytes_feature(img_name.encode()),
            # 'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtboxes_and_label.tostring()),
            'num_objects': _int64_feature(gtboxes_and_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(all_gt_label.keys()))
        count += 1

    print('\nConversion is complete!')
    writer.close()


if __name__ == '__main__':
    # xml_path = '../data/dataset/VOCdevkit/VOC2007/Annotations/000005.xml'
    # read_xml_gtbox_and_label(xml_path)

    convert_pascal_to_tfrecord()
