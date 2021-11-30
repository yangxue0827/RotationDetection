# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from alpharotate.libs.models.detectors.single_stage_base_network_batch import DetectionNetworkBase
from alpharotate.libs.models.losses.losses_fcos import LossFCOS
from alpharotate.libs.utils import nms_rotate
from alpharotate.libs.utils.coordinate_convert import backward_convert
from alpharotate.libs.models.samplers.fcos.sampler_fcos_r import SamplerFCOS


class DetectionNetworkFCOS(DetectionNetworkBase):

    def __init__(self, cfgs, is_training):
        super(DetectionNetworkFCOS, self).__init__(cfgs, is_training)
        # self.cfgs = cfgs
        # self.is_training = is_training
        self.sampler_fcos = SamplerFCOS(cfgs)
        self.losses = LossFCOS(self.cfgs)
        # self.losses_dict = {}
        # self.batch_size = cfgs.BATCH_SIZE if is_training else 1
        # self.backbone = BuildBackbone(cfgs, is_training)

    def rpn_cls_net(self, inputs, scope_list, reuse_flag, level):
        rpn_conv2d_3x3 = inputs
        for i in range(self.cfgs.NUM_SUBNET_CONV):
            rpn_conv2d_3x3 = slim.conv2d(inputs=rpn_conv2d_3x3,
                                         num_outputs=256,
                                         kernel_size=[3, 3],
                                         stride=1,
                                         activation_fn=None if self.cfgs.USE_GN else tf.nn.relu,
                                         weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                         scope='{}_{}'.format(scope_list[0], i),
                                         reuse=reuse_flag)

            if self.cfgs.USE_GN:
                rpn_conv2d_3x3 = tf.contrib.layers.group_norm(rpn_conv2d_3x3)
                rpn_conv2d_3x3 = tf.nn.relu(rpn_conv2d_3x3)

        rpn_box_scores = slim.conv2d(rpn_conv2d_3x3,
                                     num_outputs=self.cfgs.CLASS_NUM,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=self.cfgs.FINAL_CONV_WEIGHTS_INITIALIZER,
                                     biases_initializer=self.cfgs.FINAL_CONV_BIAS_INITIALIZER,
                                     scope=scope_list[2],
                                     activation_fn=None,
                                     reuse=reuse_flag)

        # rpn_box_scores = tf.reshape(rpn_box_scores, [self.batch_size, -1, self.cfgs.CLASS_NUM],
        #                             name='rpn_{}_classification_reshape'.format(level))
        rpn_box_probs = tf.nn.sigmoid(rpn_box_scores, name='rpn_{}_classification_sigmoid'.format(level))

        return rpn_box_scores, rpn_box_probs

    def rpn_reg_ctn_net(self, inputs, scope_list, reuse_flag, level):
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
                                         reuse=reuse_flag)
            if self.cfgs.USE_GN:
                rpn_conv2d_3x3 = tf.contrib.layers.group_norm(rpn_conv2d_3x3)
                rpn_conv2d_3x3 = tf.nn.relu(rpn_conv2d_3x3)

        rpn_box_offset = slim.conv2d(rpn_conv2d_3x3,
                                     num_outputs=4,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                     biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                     scope=scope_list[4]+'_offset',
                                     activation_fn=None,
                                     reuse=reuse_flag)

        # rpn_angle_sin = slim.conv2d(rpn_conv2d_3x3,
        #                             num_outputs=1,
        #                             kernel_size=[3, 3],
        #                             stride=1,
        #                             weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
        #                             biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
        #                             scope=scope_list[4] + '_sin',
        #                             activation_fn=tf.nn.sigmoid,
        #                             trainable=self.is_training,
        #                             reuse=reuse_flag)
        #
        # rpn_angle_cos = slim.conv2d(rpn_conv2d_3x3,
        #                             num_outputs=1,
        #                             kernel_size=[3, 3],
        #                             stride=1,
        #                             weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
        #                             biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
        #                             scope=scope_list[4] + '_cos',
        #                             activation_fn=tf.nn.sigmoid,
        #                             trainable=self.is_training,
        #                             reuse=reuse_flag)

        # [-90, 90]   sin in [-1, 1]  cos in [0, 1]
        # rpn_angle = (rpn_angle_sin - 0.5) * 2

        rpn_angle = slim.conv2d(rpn_conv2d_3x3,
                                num_outputs=1,
                                kernel_size=[3, 3],
                                stride=1,
                                weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                scope=scope_list[4] + '_angle',
                                activation_fn=tf.nn.sigmoid,
                                trainable=self.is_training,
                                reuse=reuse_flag)
        rpn_angle = (rpn_angle - 0.5) * 3.1415926 / 2

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

        # rpn_ctn_scores = tf.reshape(rpn_ctn_scores, [self.batch_size, -1],
        #                             name='rpn_{}_centerness_reshape'.format(level))

        # rpn_box_offset = tf.reshape(rpn_box_offset, [self.batch_size, -1, 4],
        #                             name='rpn_{}_regression_reshape'.format(level))
        return rpn_box_offset, rpn_angle, rpn_ctn_scores

    def rpn_net(self, feature_pyramid, name):

        rpn_box_offset_list = []
        rpn_box_scores_list = []
        rpn_box_probs_list = []
        rpn_cnt_scores_list = []
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):
                for level, stride in zip(self.cfgs.LEVEL, self.cfgs.ANCHOR_STRIDE):

                    if self.cfgs.SHARE_NET:
                        reuse_flag = None if level == self.cfgs.LEVEL[0] else True
                        scope_list = ['conv2d_3x3_cls', 'conv2d_3x3_reg', 'rpn_classification',
                                      'rpn_centerness', 'rpn_regression']
                    else:
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_cls_' + level, 'conv2d_3x3_reg_' + level,
                                      'rpn_classification_' + level, 'rpn_centerness' + level,
                                      'rpn_regression_' + level]

                    rpn_box_scores, rpn_box_probs = self.rpn_cls_net(feature_pyramid[level],
                                                                     scope_list, reuse_flag, level)
                    rpn_box_offset, rpn_angle, rpn_ctn_scores = self.rpn_reg_ctn_net(feature_pyramid[level],
                                                                                     scope_list, reuse_flag, level)

                    # si = tf.Variable(tf.constant(1.0),
                    #                  name='rpn_bbox_offsets_scale_'.format(level),
                    #                  dtype=tf.float32, trainable=True)
                    rpn_box_offset = tf.exp(rpn_box_offset) * stride

                    rpn_box_scores_list.append(rpn_box_scores)
                    rpn_box_probs_list.append(rpn_box_probs)
                    rpn_cnt_scores_list.append(rpn_ctn_scores)
                    rpn_box_offset_list.append(tf.concat([rpn_box_offset, rpn_angle], axis=-1))

            return rpn_box_offset_list, rpn_box_scores_list, rpn_box_probs_list, rpn_cnt_scores_list

    def _fcos_target(self, feature_pyramid, img_batch, gtboxes_batch):
        with tf.variable_scope('fcos_target'):
            fm_size_list = []
            for level in self.cfgs.LEVEL:
                featuremap_height, featuremap_width = tf.shape(feature_pyramid[level])[1], tf.shape(feature_pyramid[level])[2]
                featuremap_height = tf.cast(featuremap_height, tf.int32)
                featuremap_width = tf.cast(featuremap_width, tf.int32)
                fm_size_list.append([featuremap_height, featuremap_width])

            fcos_target_batch = tf.py_func(self.sampler_fcos.get_fcos_target_batch,
                                           inp=[gtboxes_batch, img_batch, fm_size_list],
                                           Tout=[tf.float32])
            fcos_target_batch = tf.reshape(fcos_target_batch, [self.batch_size, -1, 7])
            return fcos_target_batch

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch_h=None, gtboxes_batch_r=None, gpu_id=0):

        if self.is_training:
            # gtboxes_batch_h = tf.reshape(gtboxes_batch_h, [self.batch_size, -1, 5])
            # gtboxes_batch_h = tf.cast(gtboxes_batch_h, tf.float32)

            gtboxes_batch_r = tf.reshape(gtboxes_batch_r, [self.batch_size, -1, 9])
            gtboxes_batch_r = tf.cast(gtboxes_batch_r, tf.float32)

        if self.cfgs.USE_GN:
            input_img_batch = tf.reshape(input_img_batch, [self.batch_size, self.cfgs.IMG_SHORT_SIDE_LEN,
                                                           self.cfgs.IMG_MAX_LENGTH, 3])

        # 1. build backbone
        feature_pyramid = self.build_backbone(input_img_batch)

        # 2. build rpn
        # rpn_box_offset_list: [level, bs, h, w, 5]
        rpn_box_offset_list, rpn_cls_score_list, rpn_cls_prob_list, rpn_cnt_scores_list = self.rpn_net(feature_pyramid, 'rpn_net')

        # rpn_box_offset: [level, bs, h*w, 5]
        rpn_cls_score, rpn_cls_prob, rpn_cnt_scores, rpn_box_offset = [], [], [], []
        for i in range(len(rpn_box_offset_list)):
            rpn_cls_score.append(tf.reshape(rpn_cls_score_list[i], [self.batch_size, -1, self.cfgs.CLASS_NUM]))
            rpn_cls_prob.append(tf.reshape(rpn_cls_prob_list[i], [self.batch_size, -1, self.cfgs.CLASS_NUM]))
            rpn_box_offset.append(tf.reshape(rpn_box_offset_list[i], [self.batch_size, -1, 5]))
            rpn_cnt_scores.append(tf.reshape(rpn_cnt_scores_list[i], [self.batch_size, -1]))

        # rpn_box_offset: [bs, -1, 5]
        rpn_cls_score = tf.concat(rpn_cls_score, axis=1)
        # rpn_cls_prob = tf.concat(rpn_cls_prob, axis=1)
        rpn_cnt_scores = tf.concat(rpn_cnt_scores, axis=1)
        rpn_box_offset = tf.concat(rpn_box_offset, axis=1)

        # rpn_cnt_prob = tf.nn.sigmoid(rpn_cnt_scores)
        # rpn_cnt_prob = tf.expand_dims(rpn_cnt_prob, axis=2)
        # rpn_cnt_prob = tf.broadcast_to(rpn_cnt_prob,
        #                                [self.batch_size, tf.shape(rpn_cls_prob)[1], tf.shape(rpn_cls_prob)[2]])
        # rpn_prob = rpn_cls_prob * rpn_cnt_prob

        # 3. build loss
        if self.is_training:
            with tf.variable_scope('build_loss'):
                fcos_target_batch = self._fcos_target(feature_pyramid, input_img_batch, gtboxes_batch_r)

                cls_gt = tf.stop_gradient(fcos_target_batch[:, :, 0])
                ctr_gt = tf.stop_gradient(fcos_target_batch[:, :, 1])
                geo_gt = tf.stop_gradient(fcos_target_batch[:, :, 2:])

                cls_loss = self.losses.focal_loss_fcos(rpn_cls_score, cls_gt,
                                                       alpha=self.cfgs.ALPHA, gamma=self.cfgs.GAMMA)
                ctr_loss = self.losses.centerness_loss(rpn_cnt_scores, ctr_gt, cls_gt)
                # reg_loss = self.losses.iou_loss(geo_gt, rpn_box_offset, cls_gt, weight=ctr_gt)
                # left, bottom, right, top, theta = tf.unstack(geo_gt, axis=-1)
                # geo_gt = tf.stack([left, bottom, right, top, tf.sin(theta), tf.cos(theta)], axis=-1)
                reg_loss = self.losses.smooth_l1_loss(geo_gt, rpn_box_offset, cls_gt, weight=ctr_gt)

                self.losses_dict['cls_loss'] = cls_loss * self.cfgs.CLS_WEIGHT
                self.losses_dict['reg_loss'] = reg_loss * self.cfgs.REG_WEIGHT
                self.losses_dict['ctr_loss'] = ctr_loss * self.cfgs.CTR_WEIGHT

        # 5. postprocess
        with tf.variable_scope('postprocess_detctions'):

            boxes, scores, category = self.postprocess_detctions(rpn_box_offset_list=[rpn_box_offset[0, :, :, :] for rpn_box_offset in rpn_box_offset_list],
                                                                 rpn_cls_prob_list=[rpn_cls_prob[0, :, :, :] for rpn_cls_prob in rpn_cls_prob_list],
                                                                 rpn_cnt_scores_list=[rpn_cnt_scores[0, :, :, :] for rpn_cnt_scores in rpn_cnt_scores_list],
                                                                 gpu_id=gpu_id)
            boxes = tf.stop_gradient(boxes)
            scores = tf.stop_gradient(scores)
            category = tf.stop_gradient(category)

        if self.is_training:
            return boxes, scores, category, self.losses_dict
        else:
            return boxes, scores, category

    def postprocess_detctions(self, rpn_box_offset_list, rpn_cls_prob_list, rpn_cnt_scores_list, gpu_id):

        def get_boxes_tf(points, geometry):
            # pointx, pointy = points[:, 0], points[:, 1]
            pointx, pointy = tf.unstack(points, axis=1)
            left, bottom, right, top, theta = geometry[:, 0], geometry[:, 1], geometry[:, 2], geometry[:, 3], geometry[:, 4]
            xlt, ylt = pointx - left, pointy - top
            xlb, ylb = pointx - left, pointy + bottom
            xrb, yrb = pointx + right, pointy + bottom
            xrt, yrt = pointx + right, pointy - top

            # theta = tf.atan(sin_theta/cos_theta)
            theta *= -1

            xlt_ = tf.cos(theta) * (xlt - pointx) + tf.sin(theta) * (ylt - pointy) + pointx
            ylt_ = -tf.sin(theta) * (xlt - pointx) + tf.cos(theta) * (ylt - pointy) + pointy

            xrt_ = tf.cos(theta) * (xrt - pointx) + tf.sin(theta) * (yrt - pointy) + pointx
            yrt_ = -tf.sin(theta) * (xrt - pointx) + tf.cos(theta) * (yrt - pointy) + pointy

            xld_ = tf.cos(theta) * (xlb - pointx) + tf.sin(theta) * (ylb - pointy) + pointx
            yld_ = -tf.sin(theta) * (xlb - pointx) + tf.cos(theta) * (ylb - pointy) + pointy

            xrd_ = tf.cos(theta) * (xrb - pointx) + tf.sin(theta) * (yrb - pointy) + pointx
            yrd_ = -tf.sin(theta) * (xrb - pointx) + tf.cos(theta) * (yrb - pointy) + pointy

            convert_box = tf.transpose(tf.stack([xlt_, ylt_, xrt_, yrt_, xrd_, yrd_, xld_, yld_], axis=0))
            return convert_box

        rpn_box_offset, rpn_cnt_scores, rpn_cls_prob, center = [], [], [], []

        for i in range(len(rpn_box_offset_list)):
            shift = 0.0
            fm_height, fm_width = tf.shape(rpn_cnt_scores_list[i])[0], tf.shape(rpn_cnt_scores_list[i])[1]
            y_list = tf.linspace(tf.constant(shift), tf.cast(fm_height, tf.float32) - tf.constant(shift),
                                 tf.cast(fm_height, tf.int32))
            y_list = tf.broadcast_to(tf.reshape(y_list, [fm_height, 1, 1]), [fm_height, fm_width, 1])
            x_list = tf.linspace(tf.constant(shift), tf.cast(fm_width, tf.float32) - tf.constant(shift),
                                 tf.cast(fm_width, tf.int32))
            x_list = tf.broadcast_to(tf.reshape(x_list, [1, fm_width, 1]), [fm_height, fm_width, 1])

            xy_list = tf.concat([x_list, y_list], axis=2) * self.cfgs.ANCHOR_STRIDE[i]
            # yx_list = tf.concat([y_list, x_list], axis=2) * self.cfgs.ANCHOR_STRIDE[i]
            center.append(tf.reshape(xy_list, [-1, 2]))

            rpn_cls_prob.append(tf.reshape(rpn_cls_prob_list[i], [-1, self.cfgs.CLASS_NUM]))
            rpn_cnt_scores.append(tf.reshape(rpn_cnt_scores_list[i], [-1, ]))
            rpn_box_offset.append(tf.reshape(rpn_box_offset_list[i], [-1, 5]))
        rpn_cls_prob = tf.concat(rpn_cls_prob, axis=0)
        rpn_cnt_scores = tf.concat(rpn_cnt_scores, axis=0)
        rpn_box_offset = tf.concat(rpn_box_offset, axis=0)
        center = tf.concat(center, axis=0)

        boxes_pred = get_boxes_tf(center, rpn_box_offset)

        return_boxes_pred, return_scores, return_labels = [], [], []
        for j in range(0, self.cfgs.CLASS_NUM):
            scores = rpn_cls_prob[:, j]
            if self.is_training:
                indices = tf.reshape(tf.where(tf.greater(scores, self.cfgs.VIS_SCORE)), [-1, ])
            else:
                indices = tf.reshape(tf.where(tf.greater(scores, self.cfgs.FILTERED_SCORE)), [-1, ])

            boxes_pred = tf.gather(boxes_pred, indices)
            scores = tf.gather(scores, indices)
            rpn_cnt_scores = tf.gather(rpn_cnt_scores, indices)

            rpn_cnt_prob = tf.nn.sigmoid(rpn_cnt_scores)
            rpn_prob = scores * rpn_cnt_prob

            boxes_pred = tf.py_func(backward_convert,
                                    inp=[boxes_pred, False],
                                    Tout=[tf.float32])
            boxes_pred = tf.reshape(boxes_pred, [-1, 5])

            max_output_size = 4000 if 'DOTA' in self.cfgs.NET_NAME else 200
            nms_indices = nms_rotate.nms_rotate(decode_boxes=boxes_pred,
                                                scores=rpn_prob,
                                                iou_threshold=self.cfgs.NMS_IOU_THRESHOLD,
                                                max_output_size=100 if self.is_training else max_output_size,
                                                use_gpu=True,
                                                gpu_id=gpu_id)

            tmp_boxes_pred = tf.reshape(tf.gather(boxes_pred, nms_indices), [-1, 5])
            tmp_scores = tf.reshape(tf.gather(scores, nms_indices), [-1, ])

            return_boxes_pred.append(tmp_boxes_pred)
            return_scores.append(tmp_scores)
            return_labels.append(tf.ones_like(tmp_scores) * (j + 1))

        return_boxes_pred = tf.concat(return_boxes_pred, axis=0)
        return_scores = tf.concat(return_scores, axis=0)
        return_labels = tf.concat(return_labels, axis=0)

        return return_boxes_pred, return_scores, return_labels
