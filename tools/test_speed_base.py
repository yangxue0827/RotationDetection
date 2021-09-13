# -*- coding:utf-8 -*-
# Author: Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>
#
# License: Apache-2.0 license

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import cv2
import argparse
from tqdm import tqdm
sys.path.append("../")

from libs.label_name_dict.label_dict import LabelMap
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test Speed')
    parser.add_argument('--img_dir', dest='img_dir',
                        help='images path',
                        default='/data/dataset/HRSC2016/HRSC2016/Test/AllImages', type=str)
    parser.add_argument('--image_ext', dest='image_ext',
                        help='image format',
                        default='.bmp', type=str)
    parser.add_argument('--test_annotation_path', dest='test_annotation_path',
                        help='test annotate path',
                        default='/data/dataset/HRSC2016/HRSC2016/Test/xmls', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='', type=str)
    parser.add_argument('--draw_imgs', '-s', default=False,
                        action='store_true')
    parser.add_argument('--multi_scale', '-ms', default=False,
                        action='store_true')
    parser.add_argument('--cpu_nms', '-cn', default=False,
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class Test(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.args = parse_args()
        label_map = LabelMap(cfgs)
        self.name_label_map, self.label_name_map = label_map.name2label(), label_map.label2name()

    def eval_with_plac(self, img_dir, det_net, image_ext):

        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        # 1. preprocess img
        img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
        img_batch = tf.cast(img_plac, tf.float32)

        pretrain_zoo = PretrainModelZoo()
        if self.cfgs.NET_NAME in pretrain_zoo.pth_zoo or self.cfgs.NET_NAME in pretrain_zoo.mxnet_zoo:
            img_batch = (img_batch / 255 - tf.constant(self.cfgs.PIXEL_MEAN_)) / tf.constant(self.cfgs.PIXEL_STD)
        else:
            img_batch = img_batch - tf.constant(self.cfgs.PIXEL_MEAN)

        img_batch = tf.expand_dims(img_batch, axis=0)

        output = det_net.build_whole_detection_network(
            input_img_batch=img_batch)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer, restore_ckpt = det_net.get_restorer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            all_boxes_r = []
            imgs = os.listdir(img_dir)
            pbar = tqdm(imgs)
            for a_img_name in pbar:
                a_img_name = a_img_name.split(image_ext)[0]

                raw_img = cv2.imread(os.path.join(img_dir,
                                                  a_img_name + image_ext))
                raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

                img_short_side_len_list = self.cfgs.IMG_SHORT_SIDE_LEN if isinstance(self.cfgs.IMG_SHORT_SIDE_LEN, list) else [
                    self.cfgs.IMG_SHORT_SIDE_LEN]
                img_short_side_len_list = [img_short_side_len_list[0]] if not self.args.multi_scale else img_short_side_len_list

                for short_size in img_short_side_len_list:
                    max_len = self.cfgs.IMG_MAX_LENGTH
                    if raw_h < raw_w:
                        new_h, new_w = short_size, min(int(short_size * float(raw_w) / raw_h), max_len)
                    else:
                        new_h, new_w = min(int(short_size * float(raw_h) / raw_w), max_len), short_size
                    img_resize = cv2.resize(raw_img, (new_w, new_h))

                    output_ = \
                        sess.run(
                            [output],
                            feed_dict={img_plac: img_resize[:, :, ::-1]}
                        )

                pbar.set_description("Eval image %s" % a_img_name)

            # fw1 = open(cfgs.VERSION + '_detections_r.pkl', 'wb')
            # pickle.dump(all_boxes_r, fw1)
            return all_boxes_r


