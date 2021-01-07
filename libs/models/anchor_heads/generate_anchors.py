# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

from libs.models.anchor_heads import generate_h_anchors, generate_r_anchors, generate_h_anchors_tf


class GenerateAnchors(object):

    def __init__(self, cfgs, mode):
        self.cfgs = cfgs
        self.mode = mode

    def anchor_generator(self, base_size, feat_h, feat_w, stride, mode):
        if mode == 'H':
            anchors = tf.py_func(generate_h_anchors.generate_anchors_pre,
                                 inp=[feat_h, feat_w, stride,
                                      np.array(self.cfgs.ANCHOR_SCALES) * stride, self.cfgs.ANCHOR_RATIOS, 4.0],
                                 Tout=[tf.float32])
            anchors = tf.reshape(anchors, [-1, 4])
        else:
            anchors = generate_r_anchors.make_anchors(base_anchor_size=base_size,
                                                      anchor_scales=self.cfgs.ANCHOR_SCALES,
                                                      anchor_ratios=self.cfgs.ANCHOR_RATIOS,
                                                      anchor_angles=self.cfgs.ANCHOR_ANGLES,
                                                      featuremap_height=feat_h,
                                                      featuremap_width=feat_w,
                                                      stride=stride)
        return anchors

    def generate_all_anchor(self, feature_pyramid):

        '''
            (level, base_anchor_size) tuple:
            (P3, 32), (P4, 64), (P5, 128), (P6, 256), (P7, 512)
        '''

        anchor_list = []
        with tf.name_scope('make_anchors_all_level'):
            for level, base_size, stride in zip(self.cfgs.LEVEL, self.cfgs.BASE_ANCHOR_SIZE_LIST,
                                                self.cfgs.ANCHOR_STRIDE):
                feat_h, feat_w = tf.shape(feature_pyramid[level])[1], \
                                 tf.shape(feature_pyramid[level])[2]

                feat_h = tf.cast(feat_h, tf.float32)
                feat_w = tf.cast(feat_w, tf.float32)

                anchor_tmp = self.anchor_generator(base_size, feat_h, feat_w, stride, self.mode)
                anchor_list.append(anchor_tmp)

        return anchor_list

    def generate_all_anchor_pb(self, h_dict, w_dict):

        '''
            (level, base_anchor_size) tuple:
            (P3, 32), (P4, 64), (P5, 128), (P6, 256), (P7, 512)
        '''

        anchor_list = []
        with tf.name_scope('make_anchors_all_level'):
            for level, base_size, stride in zip(self.cfgs.LEVEL, self.cfgs.BASE_ANCHOR_SIZE_LIST,
                                                self.cfgs.ANCHOR_STRIDE):
                feat_h, feat_w = h_dict[level], w_dict[level]

                anchor_tmp = generate_h_anchors.generate_anchors_pre(int(feat_h), int(feat_w), stride,
                                                                     np.array(self.cfgs.ANCHOR_SCALES) * stride,
                                                                     self.cfgs.ANCHOR_RATIOS, 4)
                anchor_list.append(anchor_tmp)

        return anchor_list

    def generate_all_anchor_tf(self, feature_pyramid):

        '''
            (level, base_anchor_size) tuple:
            (P3, 32), (P4, 64), (P5, 128), (P6, 256), (P7, 512)
        '''

        anchor_list = []
        with tf.name_scope('make_anchors_all_level'):
            for level, base_size, stride in zip(self.cfgs.LEVEL, self.cfgs.BASE_ANCHOR_SIZE_LIST,
                                                self.cfgs.ANCHOR_STRIDE):
                feat_h, feat_w = tf.shape(feature_pyramid[level])[1], \
                                 tf.shape(feature_pyramid[level])[2]

                feat_h = tf.cast(feat_h, tf.float32)
                feat_w = tf.cast(feat_w, tf.float32)

                anchor_tmp = generate_h_anchors_tf.generate_anchors_pre(feat_h, feat_w, stride,
                                                                        np.array(self.cfgs.ANCHOR_SCALES) * stride,
                                                                        self.cfgs.ANCHOR_RATIOS, 4)
                anchor_tmp = tf.reshape(anchor_tmp, [-1, 4])
                anchor_list.append(anchor_tmp)

        return anchor_list


