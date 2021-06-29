# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.models.detectors.single_stage_base_network_batch import DetectionNetworkBase
from libs.models.losses.losses_fcos import LossFCOS
from libs.utils import bbox_transform, nms_rotate
from libs.models.samplers.fcos.sampler_fcos_h import SamplerFCOS
from libs.utils.coordinate_convert import backward_convert


class DetectionNetworkFCOS(DetectionNetworkBase):

    def __init__(self, cfgs, is_training):
        super(DetectionNetworkFCOS, self).__init__(cfgs, is_training)
        self.sampler_fcos = SamplerFCOS(cfgs)
        self.losses = LossFCOS(self.cfgs)

    def linspace(self, start, end, num):
        return np.array(np.linspace(start, end, num), np.float32)

    def get_rpn_bbox(self, offsets, stride):

        batch, fm_height, fm_width = tf.shape(offsets)[0], tf.shape(offsets)[1], tf.shape(offsets)[2]
        offsets = tf.reshape(offsets, [self.batch_size, -1, 8])

        # y_list = tf.py_func(self.linspace, inp=[tf.constant(0.5), tf.cast(fm_height, tf.float32)-tf.constant(0.5),
        #                                         tf.cast(fm_height, tf.float32)],
        #                     Tout=[tf.float32])
        y_list = tf.linspace(tf.constant(0.5), tf.cast(fm_height, tf.float32) - tf.constant(0.5),
                             tf.cast(fm_height, tf.int32))

        # y_list = tf.broadcast_to(tf.reshape(y_list, [1, fm_height, 1, 1]), [1, fm_height, fm_width, 1])
        y_list = tf.tile(tf.reshape(y_list, [1, fm_height, 1, 1]), [1, 1, fm_width, 1])

        # x_list = tf.py_func(self.linspace, inp=[tf.constant(0.5), tf.cast(fm_width, tf.float32)-tf.constant(0.5),
        #                                         tf.cast(fm_width, tf.float32)],
        #                     Tout=[tf.float32])
        x_list = tf.linspace(tf.constant(0.5), tf.cast(fm_width, tf.float32) - tf.constant(0.5),
                             tf.cast(fm_width, tf.int32))
        # x_list = tf.broadcast_to(tf.reshape(x_list, [1, 1, fm_width, 1]), [1, fm_height, fm_width, 1])
        x_list = tf.tile(tf.reshape(x_list, [1, 1, fm_width, 1]), [1, fm_height, 1, 1])

        xy_list = tf.concat([x_list, y_list], axis=3) * stride

        # center = tf.reshape(tf.broadcast_to(xy_list, [self.batch_size, fm_height, fm_width, 2]),
        #                     [self.batch_size, -1, 2])
        center = tf.reshape(tf.tile(xy_list, [self.batch_size, 1, 1, 1]), [self.batch_size, -1, 2])

        tmp_box = []
        for i in range(4):
            tmp_box.append(tf.expand_dims(center[:, :, 0] + offsets[:, :, i * 2], axis=2))
            tmp_box.append(tf.expand_dims(center[:, :, 1] + offsets[:, :, i * 2 + 1], axis=2))
        all_boxes = tf.concat(tmp_box, axis=2)
        return all_boxes

    def rpn_cls_net(self, inputs, scope_list, reuse_flag, level):
        rpn_conv2d_3x3 = inputs
        for i in range(self.cfgs.NUM_SUBNET_CONV):
            rpn_conv2d_3x3 = slim.conv2d(inputs=rpn_conv2d_3x3,
                                         num_outputs=self.cfgs.FPN_CHANNEL,
                                         kernel_size=[3, 3],
                                         stride=1,
                                         activation_fn=None if self.cfgs.USE_GN else tf.nn.relu,
                                         weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                         scope='{}_{}'.format(scope_list[0], i),
                                         trainable=self.is_training,
                                         reuse=reuse_flag)

            if self.cfgs.USE_GN:
                rpn_conv2d_3x3 = tf.contrib.layers.group_norm(rpn_conv2d_3x3)
                rpn_conv2d_3x3 = tf.nn.relu(rpn_conv2d_3x3)

        rpn_box_scores = slim.conv2d(rpn_conv2d_3x3,
                                     num_outputs=self.cfgs.CLASS_NUM,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                     biases_initializer=self.cfgs.FINAL_CONV_BIAS_INITIALIZER,
                                     scope=scope_list[2],
                                     activation_fn=None,
                                     trainable=self.is_training,
                                     reuse=reuse_flag)

        rpn_box_scores = tf.reshape(rpn_box_scores, [self.batch_size, -1, self.cfgs.CLASS_NUM],
                                    name='rpn_{}_classification_reshape'.format(level))
        rpn_box_probs = tf.nn.sigmoid(rpn_box_scores, name='rpn_{}_classification_sigmoid'.format(level))

        return rpn_box_scores, rpn_box_probs

    def rpn_reg_net(self, inputs, scope_list, reuse_flag, level):
        rpn_conv2d_3x3 = inputs
        for i in range(self.cfgs.NUM_SUBNET_CONV):
            rpn_conv2d_3x3 = slim.conv2d(inputs=rpn_conv2d_3x3,
                                         num_outputs=self.cfgs.FPN_CHANNEL,
                                         kernel_size=[3, 3],
                                         weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                         stride=1,
                                         activation_fn=None if self.cfgs.USE_GN else tf.nn.relu,
                                         scope='{}_{}'.format(scope_list[1], i),
                                         trainable=self.is_training,
                                         reuse=reuse_flag)

            if self.cfgs.USE_GN:
                rpn_conv2d_3x3 = tf.contrib.layers.group_norm(rpn_conv2d_3x3)
                rpn_conv2d_3x3 = tf.nn.relu(rpn_conv2d_3x3)

        rpn_box_offset = slim.conv2d(rpn_conv2d_3x3,
                                     num_outputs=8,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                     biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                     scope=scope_list[4],
                                     activation_fn=None,
                                     trainable=self.is_training,
                                     reuse=reuse_flag)

        rpn_ctn_scores = slim.conv2d(rpn_conv2d_3x3,
                                     num_outputs=1,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                     biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                     scope=scope_list[3],
                                     activation_fn=None,
                                     reuse=reuse_flag)
        tf.summary.image('centerness_{}'.format(level),
                         tf.nn.sigmoid(tf.expand_dims(rpn_ctn_scores[0, :, :, :], axis=0)))

        # rpn_delta_boxes = tf.reshape(rpn_delta_boxes, [self.batch_size, -1, 5],
        #                              name='rpn_{}_regression_reshape'.format(level))
        rpn_ctn_scores = tf.reshape(rpn_ctn_scores, [self.batch_size, -1],
                                    name='rpn_{}_centerness_reshape'.format(level))
        return rpn_box_offset, rpn_ctn_scores

    def rpn_net(self, feature_pyramid, name):

        rpn_box_list = []
        rpn_cls_scores_list = []
        rpn_cls_probs_list = []
        rpn_cnt_scores_list = []
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):
                for level, stride in zip(self.cfgs.LEVEL, self.cfgs.ANCHOR_STRIDE):

                    if self.cfgs.SHARE_NET:
                        reuse_flag = None if level == 'P3' else True
                        scope_list = ['conv2d_3x3_cls', 'conv2d_3x3_reg', 'rpn_classification',
                                      'rpn_centerness', 'rpn_regression']
                    else:
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_cls_' + level, 'conv2d_3x3_reg_' + level,
                                      'rpn_classification_' + level, 'rpn_centerness' + level,
                                      'rpn_regression_' + level]

                    rpn_cls_scores, rpn_cls_probs = self.rpn_cls_net(feature_pyramid[level], scope_list, reuse_flag, level)
                    rpn_box_offset, rpn_ctn_scores = self.rpn_reg_net(feature_pyramid[level], scope_list, reuse_flag, level)

                    rpn_box_offset = rpn_box_offset * stride
                    rpn_bbox = self.get_rpn_bbox(rpn_box_offset, stride)

                    rpn_cls_scores_list.append(rpn_cls_scores)
                    rpn_cls_probs_list.append(rpn_cls_probs)
                    rpn_cnt_scores_list.append(rpn_ctn_scores)
                    rpn_box_list.append(rpn_bbox)

                all_rpn_cls_scores_list = tf.concat(rpn_cls_scores_list, axis=1)
                all_rpn_cls_probs_list = tf.concat(rpn_cls_probs_list, axis=1)
                all_rpn_cnt_scores_list = tf.concat(rpn_cnt_scores_list, axis=1)
                all_rpn_box_list = tf.concat(rpn_box_list, axis=1)

            return all_rpn_cls_scores_list, all_rpn_cls_probs_list, all_rpn_cnt_scores_list, all_rpn_box_list

    def _fcos_target(self, feature_pyramid, img_batch, gtboxes_batch, gtboxes_batch_r):
        with tf.variable_scope('fcos_target'):
            fm_size_list = []
            for level in self.cfgs.LEVEL:
                featuremap_height, featuremap_width = tf.shape(feature_pyramid[level])[1], tf.shape(feature_pyramid[level])[2]
                featuremap_height = tf.cast(featuremap_height, tf.int32)
                featuremap_width = tf.cast(featuremap_width, tf.int32)
                fm_size_list.append([featuremap_height, featuremap_width])

            fcos_target_batch = tf.py_func(self.sampler_fcos.get_fcos_target_batch,
                                           inp=[gtboxes_batch, gtboxes_batch_r, img_batch, fm_size_list],
                                           Tout=[tf.float32])
            fcos_target_batch = tf.reshape(fcos_target_batch, [self.batch_size, -1, 10])
            return fcos_target_batch

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch_h=None, gtboxes_batch_r=None, gpu_id=0):

        if self.is_training:
            gtboxes_batch_h = tf.reshape(gtboxes_batch_h, [self.batch_size, -1, 5])
            gtboxes_batch_h = tf.cast(gtboxes_batch_h, tf.float32)

            gtboxes_batch_r = tf.reshape(gtboxes_batch_r, [self.batch_size, -1, 9])
            gtboxes_batch_r = tf.cast(gtboxes_batch_r, tf.float32)

        if self.cfgs.USE_GN:
            input_img_batch = tf.reshape(input_img_batch, [self.batch_size, self.cfgs.IMG_SHORT_SIDE_LEN,
                                                           self.cfgs.IMG_MAX_LENGTH, 3])

        # 1. build backbone
        feature_pyramid = self.build_backbone(input_img_batch)

        # 2. build rpn
        rpn_cls_score, rpn_cls_prob, rpn_cnt_scores, rpn_box_pred = self.rpn_net(feature_pyramid, 'rpn_net')

        rpn_cnt_prob = tf.nn.sigmoid(rpn_cnt_scores)
        rpn_cnt_prob = tf.expand_dims(rpn_cnt_prob, axis=2)
        # rpn_cnt_prob = tf.broadcast_to(rpn_cnt_prob,
        #                                [self.batch_size, tf.shape(rpn_cls_probs)[1], tf.shape(rpn_cls_probs)[2]])
        rpn_cnt_prob = tf.tile(rpn_cnt_prob, [1, 1, tf.shape(rpn_cls_prob)[2]])

        rpn_prob = rpn_cls_prob * rpn_cnt_prob
        rpn_box_pred = tf.reshape(rpn_box_pred, [self.batch_size, -1, 8])

        # 3. build loss
        if self.is_training:
            with tf.variable_scope('build_loss'):
                fcos_target_batch = self._fcos_target(feature_pyramid, input_img_batch, gtboxes_batch_h, gtboxes_batch_r)

                cls_gt = tf.stop_gradient(fcos_target_batch[:, :, 0])
                ctr_gt = tf.stop_gradient(fcos_target_batch[:, :, 1])
                gt_boxes = tf.stop_gradient(fcos_target_batch[:, :, 2:])
                cls_gt = tf.reshape(cls_gt, [self.batch_size, -1])

                rpn_cls_loss = self.losses.focal_loss_fcos(rpn_cls_score, cls_gt)

                rpn_bbox_loss = self.losses.smooth_l1_loss_fcos(gt_boxes, rpn_box_pred, cls_gt, weight=ctr_gt)

                rpn_ctr_loss = self.losses.centerness_loss(rpn_cnt_scores, ctr_gt, cls_gt)
                self.losses_dict = {
                    'rpn_cls_loss': rpn_cls_loss * self.cfgs.CLS_WEIGHT,
                    'rpn_bbox_loss': rpn_bbox_loss * self.cfgs.REG_WEIGHT,
                    'rpn_ctr_loss': rpn_ctr_loss * self.cfgs.CTR_WEIGHT
                }

        # 4. postprocess
        with tf.variable_scope('postprocess_detctions'):
            boxes, scores, category = self.postprocess_detctions(rpn_bbox_pred=rpn_box_pred[0, :, :],
                                                                 rpn_cls_prob=rpn_prob[0, :, :],
                                                                 gpu_id=gpu_id)
            boxes = tf.stop_gradient(boxes)
            scores = tf.stop_gradient(scores)
            category = tf.stop_gradient(category)

        if self.is_training:
            return boxes, scores, category, self.losses_dict
        else:
            return boxes, scores, category

    def postprocess_detctions(self, rpn_bbox_pred, rpn_cls_prob, gpu_id):

        def filter_detections(boxes, scores, is_training):

            if is_training:
                indices = tf.reshape(tf.where(tf.greater(scores, self.cfgs.VIS_SCORE)), [-1, ])
            else:
                indices = tf.reshape(tf.where(tf.greater(scores, self.cfgs.FILTERED_SCORE)), [-1, ])

            filtered_boxes = tf.gather(boxes, indices)
            filtered_scores = tf.gather(scores, indices)

            filtered_boxes = tf.py_func(func=backward_convert,
                                        inp=[filtered_boxes, False],
                                        Tout=[tf.float32])
            filtered_boxes = tf.reshape(filtered_boxes, [-1, 5])

            max_output_size = 4000 if 'DOTA' in self.cfgs.NET_NAME else 200
            nms_indices = nms_rotate.nms_rotate(decode_boxes=filtered_boxes,
                                                scores=filtered_scores,
                                                iou_threshold=self.cfgs.NMS_IOU_THRESHOLD,
                                                max_output_size=100 if self.is_training else max_output_size,
                                                use_gpu=True,
                                                gpu_id=gpu_id)

            # filter indices based on NMS
            indices = tf.gather(indices, nms_indices)

            return indices

        return_boxes_pred = []
        return_scores = []
        return_labels = []
        for j in range(0, self.cfgs.CLASS_NUM):
            indices = filter_detections(rpn_bbox_pred, rpn_cls_prob[:, j], self.is_training)
            tmp_boxes_pred = tf.reshape(tf.gather(rpn_bbox_pred, indices), [-1, 8])
            tmp_scores = tf.reshape(tf.gather(rpn_cls_prob[:, j], indices), [-1, ])

            return_boxes_pred.append(tmp_boxes_pred)
            return_scores.append(tmp_scores)
            return_labels.append(tf.ones_like(tmp_scores) * (j + 1))

        return_boxes_pred = tf.concat(return_boxes_pred, axis=0)
        return_scores = tf.concat(return_scores, axis=0)
        return_labels = tf.concat(return_labels, axis=0)

        return return_boxes_pred, return_scores, return_labels
