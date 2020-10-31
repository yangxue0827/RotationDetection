# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


class BoxHead(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def fc_head(self, roi_extractor, rois_list, feature_pyramid, img_shape, is_training, mode=0):
        with tf.variable_scope('Fast-RCNN'):

            with tf.variable_scope('rois_pooling'):
                roi_features_list = []
                for level_name, rois in zip(self.cfgs.LEVEL, rois_list):  # exclude P6_rois

                    # if mode == 0:
                    roi_features = roi_extractor.roi_align(feature_maps=feature_pyramid[level_name],
                                                           rois=rois, img_shape=img_shape,
                                                           scope=level_name)
                    # else:
                    #     raise Exception('only support roi align (mode=0)')

                    roi_features_list.append(roi_features)

                all_roi_features = tf.concat(roi_features_list, axis=0)  # [minibatch_size, H, W, C]

            with tf.variable_scope('build_fc_layers'):
                inputs = slim.flatten(inputs=all_roi_features, scope='flatten_inputs')
                fc1 = slim.fully_connected(inputs, num_outputs=1024, scope='fc1')

                fc2 = slim.fully_connected(fc1, num_outputs=1024, scope='fc2')

            with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):
                cls_score = slim.fully_connected(fc2,
                                                 num_outputs=self.cfgs.CLASS_NUM + 1,
                                                 weights_initializer=self.cfgs.INITIALIZER,
                                                 activation_fn=None, trainable=is_training,
                                                 scope='cls_fc')
                bbox_pred = slim.fully_connected(fc2,
                                                 num_outputs=(self.cfgs.CLASS_NUM + 1) * 5,
                                                 weights_initializer=self.cfgs.BBOX_INITIALIZER,
                                                 activation_fn=None, trainable=is_training,
                                                 scope='reg_fc')
                cls_score = tf.reshape(cls_score, [-1, self.cfgs.CLASS_NUM + 1])
                bbox_pred = tf.reshape(bbox_pred, [-1, 5 * (self.cfgs.CLASS_NUM + 1)])

                return bbox_pred, cls_score