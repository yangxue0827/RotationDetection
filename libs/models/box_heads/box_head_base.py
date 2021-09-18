# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.utils.coordinate_convert import get_horizen_minAreaRectangle, forward_convert


class BoxHead(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def fpn_fc_head(self, roi_extractor, rois_list, feature_pyramid, img_shape, is_training, mode=0):
        with tf.variable_scope('Fast-RCNN'):

            with tf.variable_scope('rois_pooling'):
                roi_features_list = []
                for level_name, rois in zip(self.cfgs.LEVEL, rois_list):  # exclude P6_rois

                    if mode == 1:
                        rois = tf.py_func(forward_convert,
                                          inp=[rois, False],
                                          Tout=tf.float32)
                        rois = get_horizen_minAreaRectangle(rois, False)

                    roi_features = roi_extractor.roi_align(feature_maps=feature_pyramid[level_name],
                                                           rois=rois, img_shape=img_shape,
                                                           scope=level_name)
                    # else:
                    #     raise Exception('only support roi align (mode=0)')

                    roi_features_list.append(roi_features)

                all_roi_features = tf.concat(roi_features_list, axis=0)  # [minibatch_size, H, W, C]

            with tf.variable_scope('build_fc_layers'):
                inputs = slim.flatten(inputs=all_roi_features, scope='flatten_inputs')
                fc1 = slim.fully_connected(inputs, num_outputs=1024, trainable=is_training, scope='fc1')

                fc2 = slim.fully_connected(fc1, num_outputs=1024, trainable=is_training, scope='fc2')

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

    def fc_head(self, roi_extractor, rois, feature, img_shape, is_training, mode=0):
        with tf.variable_scope('Fast-RCNN'):

            with tf.variable_scope('rois_pooling'):

                # if mode == 0:
                all_roi_features = roi_extractor.roi_align(feature_maps=feature,
                                                           rois=rois,
                                                           img_shape=img_shape,
                                                           scope='')
                # else:
                #     raise Exception('only support roi align (mode=0)')

            with tf.variable_scope('build_fc_layers'):
                inputs = slim.flatten(inputs=all_roi_features, scope='flatten_inputs')
                fc1 = slim.fully_connected(inputs, num_outputs=1024, trainable=is_training, scope='fc1')

                fc2 = slim.fully_connected(fc1, num_outputs=1024, trainable=is_training, scope='fc2')

            with tf.variable_scope('horizen_branch'):
                with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):
                    cls_score_h = slim.fully_connected(fc2,
                                                       num_outputs=self.cfgs.CLASS_NUM + 1,
                                                       weights_initializer=self.cfgs.INITIALIZER,
                                                       activation_fn=None, trainable=is_training,
                                                       scope='cls_fc_h')

                    bbox_pred_h = slim.fully_connected(fc2,
                                                       num_outputs=(self.cfgs.CLASS_NUM + 1) * 4,
                                                       weights_initializer=self.cfgs.BBOX_INITIALIZER,
                                                       activation_fn=None, trainable=is_training,
                                                       scope='reg_fc_h')

                    # for convient. It also produce (cls_num +1) bboxes

                    cls_score_h = tf.reshape(cls_score_h, [-1, self.cfgs.CLASS_NUM + 1])
                    bbox_pred_h = tf.reshape(bbox_pred_h, [-1, 4 * (self.cfgs.CLASS_NUM + 1)])

            with tf.variable_scope('rotation_branch'):
                with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):
                    cls_score_r = slim.fully_connected(fc2,
                                                       num_outputs=self.cfgs.CLASS_NUM + 1,
                                                       weights_initializer=self.cfgs.INITIALIZER,
                                                       activation_fn=None, trainable=is_training,
                                                       scope='cls_fc_r')

                    bbox_pred_r = slim.fully_connected(fc2,
                                                       num_outputs=(self.cfgs.CLASS_NUM + 1) * 5,
                                                       weights_initializer=self.cfgs.BBOX_INITIALIZER,
                                                       activation_fn=None, trainable=is_training,
                                                       scope='reg_fc_r')
                    # for convient. It also produce (cls_num +1) bboxes
                    cls_score_r = tf.reshape(cls_score_r, [-1, self.cfgs.CLASS_NUM + 1])
                    bbox_pred_r = tf.reshape(bbox_pred_r, [-1, 5 * (self.cfgs.CLASS_NUM + 1)])

            return bbox_pred_h, cls_score_h, bbox_pred_r, cls_score_r

    def fpn_fc_sigmoid_head(self, roi_extractor, rois_list, feature_pyramid, img_shape, is_training, mode=0):
        with tf.variable_scope('Fast-RCNN'):

            with tf.variable_scope('rois_pooling'):
                roi_features_list = []
                for level_name, rois in zip(self.cfgs.LEVEL, rois_list):  # exclude P6_rois

                    if mode == 1:
                        rois = tf.py_func(forward_convert,
                                          inp=[rois, False],
                                          Tout=tf.float32)
                        rois = get_horizen_minAreaRectangle(rois, False)

                    roi_features = roi_extractor.roi_align(feature_maps=feature_pyramid[level_name],
                                                           rois=rois, img_shape=img_shape,
                                                           scope=level_name)
                    # else:
                    #     raise Exception('only support roi align (mode=0)')

                    roi_features_list.append(roi_features)

                all_roi_features = tf.concat(roi_features_list, axis=0)  # [minibatch_size, H, W, C]

            with tf.variable_scope('build_fc_layers'):
                inputs = slim.flatten(inputs=all_roi_features, scope='flatten_inputs')
                fc1 = slim.fully_connected(inputs, num_outputs=1024, trainable=is_training, scope='fc1')

                fc2 = slim.fully_connected(fc1, num_outputs=1024, trainable=is_training, scope='fc2')

            with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):
                cls_score = slim.fully_connected(fc2,
                                                 num_outputs=self.cfgs.CLASS_NUM,
                                                 weights_initializer=self.cfgs.INITIALIZER,
                                                 activation_fn=None, trainable=is_training,
                                                 scope='cls_fc')
                bbox_pred = slim.fully_connected(fc2,
                                                 num_outputs=5,
                                                 weights_initializer=self.cfgs.BBOX_INITIALIZER,
                                                 activation_fn=None, trainable=is_training,
                                                 scope='reg_fc')

                cls_score = tf.reshape(cls_score, [-1, self.cfgs.CLASS_NUM])
                bbox_pred = tf.reshape(bbox_pred, [-1, 5])

                return bbox_pred, cls_score

    def fpn_fc_head_cls(self, roi_extractor, rois_list, feature_pyramid, img_shape, is_training, coding_len):
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
                fc1 = slim.fully_connected(inputs, num_outputs=1024, trainable=is_training, scope='fc1')

                fc2 = slim.fully_connected(fc1, num_outputs=1024, trainable=is_training, scope='fc2')

            with slim.arg_scope([slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):
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

                angle_cls = slim.fully_connected(fc2,
                                                 num_outputs=coding_len,
                                                 weights_initializer=self.cfgs.INITIALIZER,
                                                 activation_fn=None, trainable=is_training,
                                                 scope='angle_cls_fc')

                cls_score = tf.reshape(cls_score, [-1, self.cfgs.CLASS_NUM + 1])
                angle_cls = tf.reshape(angle_cls, [-1, coding_len])
                bbox_pred = tf.reshape(bbox_pred, [-1, 5 * (self.cfgs.CLASS_NUM + 1)])

                return bbox_pred, cls_score, angle_cls
