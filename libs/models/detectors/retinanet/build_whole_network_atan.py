# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.models.detectors.single_stage_base_network import DetectionNetworkBase
from libs.models.losses.losses import Loss
from libs.utils import bbox_transform, nms_rotate
from libs.utils.coordinate_convert import coordinate_present_convert
from libs.models.samplers.retinanet.anchor_sampler_retinenet import AnchorSamplerRetinaNet


class DetectionNetworkRetinaNet(DetectionNetworkBase):

    def __init__(self, cfgs, is_training):
        super(DetectionNetworkRetinaNet, self).__init__(cfgs, is_training)
        self.anchor_sampler_retinenet = AnchorSamplerRetinaNet(cfgs)
        self.losses = Loss(self.cfgs)

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

        rpn_delta_boxes = slim.conv2d(rpn_conv2d_3x3,
                                      num_outputs=4 * self.num_anchors_per_location,
                                      kernel_size=[3, 3],
                                      stride=1,
                                      weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                      biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                      scope=scope_list[3],
                                      activation_fn=None,
                                      trainable=self.is_training,
                                      reuse=reuse_flag)

        rpn_angle_sin = slim.conv2d(rpn_conv2d_3x3,
                                    num_outputs=self.num_anchors_per_location,
                                    kernel_size=[3, 3],
                                    stride=1,
                                    weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                    biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                    scope=scope_list[3] + '_sin',
                                    activation_fn=tf.nn.sigmoid,
                                    trainable=self.is_training,
                                    reuse=reuse_flag)

        rpn_angle_cos = slim.conv2d(rpn_conv2d_3x3,
                                    num_outputs=self.num_anchors_per_location,
                                    kernel_size=[3, 3],
                                    stride=1,
                                    weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                    biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                    scope=scope_list[3] + '_cos',
                                    activation_fn=tf.nn.sigmoid,
                                    trainable=self.is_training,
                                    reuse=reuse_flag)

        if self.cfgs.ANGLE_RANGE == 180:
            # [-90, 90]   sin in [-1, 1]  cos in [0, 1]
            rpn_angle_sin = 2 * (rpn_angle_sin - 0.5)
            # [-90, 90]   sin in [-1, 1]  cos in [-1, 1]
            # rpn_angle_sin, rpn_angle_cos = 2 * (rpn_angle_sin - 0.5), 2 * (rpn_angle_cos - 0.5)
        else:
            # [-90, 0]   sin in [-1, 0]   cos in [0, 1]
            rpn_angle_sin *= -1  # not work due to period mismatch (presumably)

        rpn_angle_sin = rpn_angle_sin / tf.sqrt(tf.pow(rpn_angle_sin, 2) + tf.pow(rpn_angle_cos, 2))
        rpn_angle_cos = rpn_angle_cos / tf.sqrt(tf.pow(rpn_angle_sin, 2) + tf.pow(rpn_angle_cos, 2))

        rpn_delta_boxes = tf.reshape(rpn_delta_boxes, [-1, 4], name='rpn_{}_regression_reshape'.format(level))
        rpn_angle_sin = tf.reshape(rpn_angle_sin, [-1, 1], name='rpn_{}_sin_reshape'.format(level))
        rpn_angle_cos = tf.reshape(rpn_angle_cos, [-1, 1], name='rpn_{}_cos_reshape'.format(level))

        rpn_delta_boxes = tf.concat([rpn_delta_boxes, rpn_angle_sin, rpn_angle_cos], axis=-1)

        return rpn_delta_boxes

    def rpn_net(self, feature_pyramid, name):

        rpn_delta_boxes_list = []
        rpn_scores_list = []
        rpn_probs_list = []
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):
                for level in self.cfgs.LEVEL:

                    if self.cfgs.SHARE_NET:
                        reuse_flag = None if level == self.cfgs.LEVEL[0] else True
                        scope_list = ['conv2d_3x3_cls', 'conv2d_3x3_reg', 'rpn_classification', 'rpn_regression']
                    else:
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_cls_' + level, 'conv2d_3x3_reg_' + level,
                                      'rpn_classification_' + level, 'rpn_regression_' + level]

                    rpn_box_scores, rpn_box_probs = self.rpn_cls_net(feature_pyramid[level], scope_list, reuse_flag, level)
                    rpn_delta_boxes = self.rpn_reg_net(feature_pyramid[level], scope_list, reuse_flag, level)

                    rpn_scores_list.append(rpn_box_scores)
                    rpn_probs_list.append(rpn_box_probs)
                    rpn_delta_boxes_list.append(rpn_delta_boxes)

                # rpn_all_delta_boxes = tf.concat(rpn_delta_boxes_list, axis=0)
                # rpn_all_boxes_scores = tf.concat(rpn_scores_list, axis=0)
                # rpn_all_boxes_probs = tf.concat(rpn_probs_list, axis=0)

            return rpn_delta_boxes_list, rpn_scores_list, rpn_probs_list

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch_h=None, gtboxes_batch_r=None, gpu_id=0):

        if self.is_training:
            gtboxes_batch_h = tf.reshape(gtboxes_batch_h, [-1, 5])
            gtboxes_batch_h = tf.cast(gtboxes_batch_h, tf.float32)

            gtboxes_batch_r = tf.reshape(gtboxes_batch_r, [-1, 6])
            gtboxes_batch_r = tf.cast(gtboxes_batch_r, tf.float32)

        if self.cfgs.USE_GN:
            input_img_batch = tf.reshape(input_img_batch, [1, self.cfgs.IMG_SHORT_SIDE_LEN,
                                                           self.cfgs.IMG_MAX_LENGTH, 3])

        # 1. build backbone
        feature_pyramid = self.build_backbone(input_img_batch)

        # 2. build rpn
        rpn_box_pred_list, rpn_cls_score_list, rpn_cls_prob_list = self.rpn_net(feature_pyramid, 'rpn_net')
        rpn_box_pred = tf.concat(rpn_box_pred_list, axis=0)
        rpn_cls_score = tf.concat(rpn_cls_score_list, axis=0)
        rpn_cls_prob = tf.concat(rpn_cls_prob_list, axis=0)

        # 3. generate anchors
        anchor_list = self.make_anchors(feature_pyramid)
        anchors = tf.concat(anchor_list, axis=0)

        # 4. build loss
        if self.is_training:
            with tf.variable_scope('build_loss'):
                labels, target_delta, anchor_states, target_boxes = tf.py_func(func=self.anchor_sampler_retinenet.anchor_target_layer,
                                                                               inp=[gtboxes_batch_h,
                                                                                    gtboxes_batch_r, anchors, gpu_id],
                                                                               Tout=[tf.float32, tf.float32, tf.float32,
                                                                                     tf.float32])

                if self.method == 'H':
                    self.add_anchor_img_smry(input_img_batch, anchors, anchor_states, 0)
                else:
                    self.add_anchor_img_smry(input_img_batch, anchors, anchor_states, 1)

                cls_loss = self.losses.focal_loss(labels, rpn_cls_score, anchor_states)

                reg_xywh_loss = self.losses.smooth_l1_loss(target_delta[:, :-1], rpn_box_pred[:, :-2], anchor_states)
                target_theta = tf.reshape(target_boxes, [-1, 6])[:, -2] + 90. if self.cfgs.ANGLE_RANGE == 180 else 0.
                target_theta = target_theta * 3.1415926 / 180.
                target_theta_sin = tf.reshape(tf.sin(target_theta), [-1, 1])
                target_theta_cos = tf.reshape(tf.cos(target_theta), [-1, 1])
                reg_theta_loss = self.losses.smooth_l1_loss(tf.concat([target_theta_sin, target_theta_cos], axis=-1), rpn_box_pred[:, -2:], anchor_states)

                reg_loss = reg_xywh_loss + reg_theta_loss

                self.losses_dict['cls_loss'] = cls_loss * self.cfgs.CLS_WEIGHT
                self.losses_dict['reg_loss'] = reg_loss * self.cfgs.REG_WEIGHT

        # 5. postprocess
        with tf.variable_scope('postprocess_detctions'):
            boxes, scores, category = self.postprocess_detctions(rpn_bbox_pred=rpn_box_pred,
                                                                 rpn_cls_prob=rpn_cls_prob,
                                                                 anchors=anchors,
                                                                 gpu_id=gpu_id)
            boxes = tf.stop_gradient(boxes)
            scores = tf.stop_gradient(scores)
            category = tf.stop_gradient(category)

        if self.is_training:
            return boxes, scores, category, self.losses_dict
        else:
            return boxes, scores, category

    def postprocess_detctions(self, rpn_bbox_pred, rpn_cls_prob, anchors, gpu_id):

        return_boxes_pred = []
        return_scores = []
        return_labels = []
        for j in range(0, self.cfgs.CLASS_NUM):
            scores = rpn_cls_prob[:, j]
            if self.is_training:
                indices = tf.reshape(tf.where(tf.greater(scores, self.cfgs.VIS_SCORE)), [-1, ])
            else:
                indices = tf.reshape(tf.where(tf.greater(scores, self.cfgs.FILTERED_SCORE)), [-1, ])

            anchors_ = tf.gather(anchors, indices)
            rpn_bbox_pred_ = tf.gather(rpn_bbox_pred, indices)
            scores = tf.gather(scores, indices)

            if self.method == 'H':
                x_c = (anchors_[:, 2] + anchors_[:, 0]) / 2
                y_c = (anchors_[:, 3] + anchors_[:, 1]) / 2
                h = anchors_[:, 2] - anchors_[:, 0] + 1
                w = anchors_[:, 3] - anchors_[:, 1] + 1
                theta = -90 * tf.ones_like(x_c)
                anchors_ = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

            if self.cfgs.ANGLE_RANGE == 180:
                anchors_ = tf.py_func(coordinate_present_convert,
                                      inp=[anchors_, -1],
                                      Tout=[tf.float32])
                anchors_ = tf.reshape(anchors_, [-1, 5])

            boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors_, deltas=rpn_bbox_pred_)

            x, y, w, h, _ = tf.unstack(boxes_pred, axis=1)
            theta = tf.atan(rpn_bbox_pred_[:, -2]/rpn_bbox_pred_[:, -1]) * 180 / 3.1415926
            boxes_pred = tf.transpose(tf.stack([x, y, w, h, theta]))

            if self.cfgs.ANGLE_RANGE == 180:

                boxes_pred = tf.py_func(coordinate_present_convert,
                                        inp=[boxes_pred, 1, False],
                                        Tout=[tf.float32])
                boxes_pred = tf.reshape(boxes_pred, [-1, 5])

            # max_output_size = 4000 if 'DOTA' in self.cfgs.NET_NAME else 200
            max_output_size = 100
            nms_indices = nms_rotate.nms_rotate(decode_boxes=boxes_pred,
                                                scores=scores,
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
