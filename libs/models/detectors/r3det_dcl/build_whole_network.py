# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.models.detectors.single_stage_base_network import DetectionNetworkBase
from libs.models.losses.losses_dcl import LossDCL
from libs.utils import bbox_transform, nms_rotate
from libs.models.samplers.r3det_dcl.anchor_sampler_r3det_dcl import AnchorSamplerR3DetDCL
from libs.models.samplers.r3det_dcl.refine_anchor_sampler_r3det_dcl import RefineAnchorSamplerR3DetDCL
from utils.densely_coded_label import get_code_len, angle_label_decode
from libs.utils.coordinate_convert import coordinate_present_convert


class DetectionNetworkR3DetDCL(DetectionNetworkBase):

    def __init__(self, cfgs, is_training):
        super(DetectionNetworkR3DetDCL, self).__init__(cfgs, is_training)
        self.anchor_sampler_retinenet = AnchorSamplerR3DetDCL(cfgs)
        self.refine_anchor_sampler_r3det_dcl = RefineAnchorSamplerR3DetDCL(cfgs)
        self.losses = LossDCL(self.cfgs)
        self.coding_len = get_code_len(int(cfgs.ANGLE_RANGE / cfgs.OMEGA), mode=cfgs.ANGLE_MODE)

    def refine_cls_net(self, inputs, scope_list, reuse_flag, level):
        rpn_conv2d_3x3 = inputs
        for i in range(self.cfgs.NUM_SUBNET_CONV):
            rpn_conv2d_3x3 = slim.conv2d(inputs=rpn_conv2d_3x3,
                                         num_outputs=self.cfgs.FPN_CHANNEL,
                                         kernel_size=[3, 3],
                                         stride=1,
                                         activation_fn=None if self.cfgs.USE_GN else tf.nn.relu,
                                         weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                         trainable=self.is_training,
                                         scope='{}_{}'.format(scope_list[0], i),
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
                                     trainable=self.is_training,
                                     scope=scope_list[2],
                                     activation_fn=None,
                                     reuse=reuse_flag)

        rpn_box_scores = tf.reshape(rpn_box_scores, [-1, self.cfgs.CLASS_NUM],
                                    name='refine_{}_classification_reshape'.format(level))
        rpn_box_probs = tf.sigmoid(rpn_box_scores, name='refine_{}_classification_sigmoid'.format(level))

        return rpn_box_scores, rpn_box_probs

    def refine_reg_net(self, inputs, scope_list, reuse_flag, level):
        rpn_conv2d_3x3 = inputs
        for i in range(self.cfgs.NUM_SUBNET_CONV):
            rpn_conv2d_3x3 = slim.conv2d(inputs=rpn_conv2d_3x3,
                                         num_outputs=self.cfgs.FPN_CHANNEL,
                                         kernel_size=[3, 3],
                                         weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                         stride=1,
                                         activation_fn=None if self.cfgs.USE_GN else tf.nn.relu,
                                         trainable=self.is_training,
                                         scope='{}_{}'.format(scope_list[1], i),
                                         reuse=reuse_flag)

            if self.cfgs.USE_GN:
                rpn_conv2d_3x3 = tf.contrib.layers.group_norm(rpn_conv2d_3x3)
                rpn_conv2d_3x3 = tf.nn.relu(rpn_conv2d_3x3)

        rpn_delta_boxes = slim.conv2d(rpn_conv2d_3x3,
                                      num_outputs=4,
                                      kernel_size=[3, 3],
                                      stride=1,
                                      weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                      biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                      trainable=self.is_training,
                                      scope=scope_list[3],
                                      activation_fn=None,
                                      reuse=reuse_flag)

        rpn_angle_cls = slim.conv2d(rpn_conv2d_3x3,
                                    num_outputs=self.coding_len,
                                    kernel_size=[3, 3],
                                    stride=1,
                                    weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                    biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                    trainable=self.is_training,
                                    scope=scope_list[4],
                                    activation_fn=None,
                                    reuse=reuse_flag)

        rpn_delta_boxes = tf.reshape(rpn_delta_boxes, [-1, 4],
                                     name='refine_{}_regression_reshape'.format(level))
        rpn_angle_cls = tf.reshape(rpn_angle_cls, [-1, self.coding_len],
                                   name='rpn_{}_angle_cls_reshape'.format(level))
        return rpn_delta_boxes, rpn_angle_cls

    def refine_net(self, feature_pyramid, name):

        refine_delta_boxes_list = []
        refine_scores_list = []
        refine_probs_list = []
        refine_angle_cls_list = []
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):
                for level in self.cfgs.LEVEL:

                    if self.cfgs.SHARE_NET:
                        reuse_flag = None if level == 'P3' else True
                        scope_list = ['conv2d_3x3_cls', 'conv2d_3x3_reg', 'refine_classification',
                                      'refine_regression', 'refine_angle_cls']
                    else:
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_cls_' + level, 'conv2d_3x3_reg_' + level,
                                      'refine_classification_' + level, 'refine_regression_' + level,
                                      'refine_angle_cls_' + level]

                    refine_box_scores, refine_box_probs = self.refine_cls_net(feature_pyramid[level],
                                                                              scope_list, reuse_flag, level)
                    refine_delta_boxes, refine_angle_cls = self.refine_reg_net(feature_pyramid[level], scope_list, reuse_flag, level)

                    refine_scores_list.append(refine_box_scores)
                    refine_probs_list.append(refine_box_probs)
                    refine_delta_boxes_list.append(refine_delta_boxes)
                    refine_angle_cls_list.append(refine_angle_cls)

            return refine_delta_boxes_list, refine_scores_list, refine_probs_list, refine_angle_cls_list

    def refine_feature_op(self, points, feature_map):

        h, w = tf.cast(tf.shape(feature_map)[1], tf.int32), tf.cast(tf.shape(feature_map)[2], tf.int32)

        xmin = tf.maximum(0.0, tf.floor(points[:, 0]))
        xmin = tf.minimum(tf.cast(w - 1, tf.float32), tf.ceil(xmin))

        ymin = tf.maximum(0.0, tf.floor(points[:, 1]))
        ymin = tf.minimum(tf.cast(h - 1, tf.float32), tf.ceil(ymin))

        xmax = tf.minimum(tf.cast(w - 1, tf.float32), tf.ceil(points[:, 0]))
        xmax = tf.maximum(0.0, tf.floor(xmax))

        ymax = tf.minimum(tf.cast(h - 1, tf.float32), tf.ceil(points[:, 1]))
        ymax = tf.maximum(0.0, tf.floor(ymax))

        left_top = tf.cast(tf.transpose(tf.stack([ymin, xmin], axis=0)), tf.int32)
        right_bottom = tf.cast(tf.transpose(tf.stack([ymax, xmax], axis=0)), tf.int32)
        left_bottom = tf.cast(tf.transpose(tf.stack([ymax, xmin], axis=0)), tf.int32)
        right_top = tf.cast(tf.transpose(tf.stack([ymin, xmax], axis=0)), tf.int32)

        feature = feature_map

        left_top_feature = tf.gather_nd(tf.squeeze(feature), left_top)
        right_bottom_feature = tf.gather_nd(tf.squeeze(feature), right_bottom)
        left_bottom_feature = tf.gather_nd(tf.squeeze(feature), left_bottom)
        right_top_feature = tf.gather_nd(tf.squeeze(feature), right_top)

        refine_feature = right_bottom_feature * tf.tile(
            tf.reshape((tf.abs((points[:, 0] - xmin) * (points[:, 1] - ymin))), [-1, 1]),
            [1, self.cfgs.FPN_CHANNEL]) \
                         + left_top_feature * tf.tile(
            tf.reshape((tf.abs((xmax - points[:, 0]) * (ymax - points[:, 1]))), [-1, 1]),
            [1, self.cfgs.FPN_CHANNEL]) \
                         + right_top_feature * tf.tile(
            tf.reshape((tf.abs((points[:, 0] - xmin) * (ymax - points[:, 1]))), [-1, 1]),
            [1, self.cfgs.FPN_CHANNEL]) \
                         + left_bottom_feature * tf.tile(
            tf.reshape((tf.abs((xmax - points[:, 0]) * (points[:, 1] - ymin))), [-1, 1]),
            [1, self.cfgs.FPN_CHANNEL])

        refine_feature = tf.reshape(refine_feature, [1, tf.cast(h, tf.int32), tf.cast(w, tf.int32), self.cfgs.FPN_CHANNEL])

        # refine_feature = tf.reshape(refine_feature, [1, tf.cast(feature_size[1], tf.int32),
        #                                              tf.cast(feature_size[0], tf.int32), 256])

        return refine_feature + feature

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch_h=None, gtboxes_batch_r=None, gt_encode_label=None, gpu_id=0):

        if self.is_training:
            gtboxes_batch_h = tf.reshape(gtboxes_batch_h, [-1, 5])
            gtboxes_batch_h = tf.cast(gtboxes_batch_h, tf.float32)

            gtboxes_batch_r = tf.reshape(gtboxes_batch_r, [-1, 6])
            gtboxes_batch_r = tf.cast(gtboxes_batch_r, tf.float32)

            gt_encode_label = tf.reshape(gt_encode_label, [-1, self.coding_len])
            gt_encode_label = tf.cast(gt_encode_label, tf.float32)

        if self.cfgs.USE_GN:
            input_img_batch = tf.reshape(input_img_batch, [1, self.cfgs.IMG_SHORT_SIDE_LEN,
                                                           self.cfgs.IMG_MAX_LENGTH, 3])

        # 1. build backbone
        feature_pyramid = self.build_backbone(input_img_batch)

        # 2. build rpn
        rpn_box_pred_list, rpn_cls_score_list, rpn_cls_prob_list = self.rpn_net(feature_pyramid, 'rpn_net')
        rpn_box_pred = tf.concat(rpn_box_pred_list, axis=0)
        rpn_cls_score = tf.concat(rpn_cls_score_list, axis=0)
        # rpn_cls_prob = tf.concat(rpn_cls_prob_list, axis=0)

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

                if self.cfgs.USE_IOU_FACTOR:
                    reg_loss = self.losses.iou_smooth_l1_loss_exp(target_delta, rpn_box_pred, anchor_states,
                                                                  target_boxes, anchors)
                else:
                    reg_loss = self.losses.smooth_l1_loss(target_delta, rpn_box_pred, anchor_states)

                self.losses_dict['cls_loss'] = cls_loss * self.cfgs.CLS_WEIGHT
                self.losses_dict['reg_loss'] = reg_loss * self.cfgs.REG_WEIGHT

        with tf.variable_scope('refine_feature_pyramid'):
            refine_feature_pyramid = {}
            refine_boxes_list = []

            for box_pred, cls_prob, anchor, stride, level in \
                    zip(rpn_box_pred_list, rpn_cls_prob_list, anchor_list,
                        self.cfgs.ANCHOR_STRIDE, self.cfgs.LEVEL):

                box_pred = tf.reshape(box_pred, [-1, self.num_anchors_per_location, 5])
                anchor = tf.reshape(anchor, [-1, self.num_anchors_per_location, 5 if self.method == 'R' else 4])
                cls_prob = tf.reshape(cls_prob, [-1, self.num_anchors_per_location, self.cfgs.CLASS_NUM])

                cls_max_prob = tf.reduce_max(cls_prob, axis=-1)
                box_pred_argmax = tf.cast(tf.reshape(tf.argmax(cls_max_prob, axis=-1), [-1, 1]), tf.int32)
                indices = tf.cast(tf.cumsum(tf.ones_like(box_pred_argmax), axis=0), tf.int32) - tf.constant(1, tf.int32)
                indices = tf.concat([indices, box_pred_argmax], axis=-1)

                box_pred_filter = tf.reshape(tf.gather_nd(box_pred, indices), [-1, 5])
                anchor_filter = tf.reshape(tf.gather_nd(anchor, indices), [-1, 5 if self.method == 'R' else 4])

                if self.cfgs.METHOD == 'H':
                    x_c = (anchor_filter[:, 2] + anchor_filter[:, 0]) / 2
                    y_c = (anchor_filter[:, 3] + anchor_filter[:, 1]) / 2
                    h = anchor_filter[:, 2] - anchor_filter[:, 0] + 1
                    w = anchor_filter[:, 3] - anchor_filter[:, 1] + 1
                    theta = -90 * tf.ones_like(x_c)
                    anchor_filter = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

                boxes_filter = bbox_transform.rbbox_transform_inv(boxes=anchor_filter, deltas=box_pred_filter)
                refine_boxes_list.append(boxes_filter)
                center_point = boxes_filter[:, :2] / stride

                refine_feature_pyramid[level] = self.refine_feature_op(points=center_point,
                                                                       feature_map=feature_pyramid[level])

        refine_box_pred_list, refine_cls_score_list, refine_cls_prob_list, refine_angle_cls_list = self.refine_net(refine_feature_pyramid, 'refine_net')

        refine_box_pred = tf.concat(refine_box_pred_list, axis=0)
        refine_cls_score = tf.concat(refine_cls_score_list, axis=0)
        refine_cls_prob = tf.concat(refine_cls_prob_list, axis=0)
        refine_angle_cls = tf.concat(refine_angle_cls_list, axis=0)
        refine_boxes = tf.concat(refine_boxes_list, axis=0)

        # 4. postprocess rpn proposals. such as: decode, clip, filter
        if self.is_training:
            with tf.variable_scope('build_refine_loss'):
                refine_labels, refine_target_delta, refine_box_states, refine_target_boxes, refine_target_encode_label = tf.py_func(
                    func=self.refine_anchor_sampler_r3det_dcl.refine_anchor_target_layer,
                    inp=[gtboxes_batch_r, gt_encode_label, refine_boxes,
                         self.cfgs.REFINE_IOU_POSITIVE_THRESHOLD[0], self.cfgs.REFINE_IOU_NEGATIVE_THRESHOLD[0], gpu_id],
                    Tout=[tf.float32, tf.float32, tf.float32,
                          tf.float32, tf.float32])

                self.add_anchor_img_smry(input_img_batch, refine_boxes, refine_box_states, 1)

                refine_cls_loss = self.losses.focal_loss(refine_labels, refine_cls_score, refine_box_states)
                refine_reg_loss = self.losses.smooth_l1_loss(refine_target_delta, refine_box_pred, refine_box_states)
                angle_cls_loss = self.losses.angle_cls_period_focal_loss(refine_target_encode_label, refine_angle_cls,
                                                                         refine_box_states, refine_target_boxes,
                                                                         decimal_weight=self.cfgs.DATASET_NAME.startswith('DOTA'))

                self.losses_dict['refine_cls_loss'] = refine_cls_loss * self.cfgs.CLS_WEIGHT
                self.losses_dict['refine_reg_loss'] = refine_reg_loss * self.cfgs.REG_WEIGHT
                self.losses_dict['angle_cls_loss'] = angle_cls_loss * self.cfgs.ANGLE_WEIGHT

        # 5. postprocess
        with tf.variable_scope('postprocess_detctions'):
            scores, category, boxes_angle = self.postprocess_detctions(refine_bbox_pred=refine_box_pred,
                                                                       refine_cls_prob=refine_cls_prob,
                                                                       refine_angle_prob=tf.sigmoid(refine_angle_cls),
                                                                       refine_boxes=refine_boxes,
                                                                       gpu_id=gpu_id)
            scores = tf.stop_gradient(scores)
            category = tf.stop_gradient(category)
            boxes_angle = tf.stop_gradient(boxes_angle)

        if self.is_training:
            return boxes_angle, scores, category, self.losses_dict
        else:
            return boxes_angle, scores, category

    def postprocess_detctions(self, refine_bbox_pred, refine_cls_prob, refine_angle_prob, refine_boxes, gpu_id):

        # return_boxes_pred = []
        return_boxes_pred_angle = []
        return_scores = []
        return_labels = []
        for j in range(0, self.cfgs.CLASS_NUM):
            scores = refine_cls_prob[:, j]
            if self.is_training:
                indices = tf.reshape(tf.where(tf.greater(scores, self.cfgs.VIS_SCORE)), [-1, ])
            else:
                indices = tf.reshape(tf.where(tf.greater(scores, self.cfgs.FILTERED_SCORE)), [-1, ])

            refine_boxes_ = tf.gather(refine_boxes, indices)
            refine_bbox_pred_ = tf.gather(refine_bbox_pred, indices)
            scores = tf.gather(scores, indices)
            refine_angle_prob_ = tf.gather(refine_angle_prob, indices)

            angle_cls = tf.py_func(angle_label_decode,
                                   inp=[refine_angle_prob_, self.cfgs.ANGLE_RANGE, self.cfgs.OMEGA, self.cfgs.ANGLE_MODE],
                                   Tout=[tf.float32])
            angle_cls = tf.reshape(angle_cls, [-1, ]) * -1

            if self.cfgs.ANGLE_RANGE == 180:
                refine_boxes_ = tf.py_func(coordinate_present_convert,
                                           inp=[refine_boxes_, -1],
                                           Tout=[tf.float32])
                refine_boxes_ = tf.reshape(refine_boxes_, [-1, 5])

            refine_boxes_pred = bbox_transform.rbbox_transform_inv_dcl(boxes=refine_boxes_, deltas=refine_bbox_pred_)
            refine_boxes_pred = tf.reshape(refine_boxes_pred, [-1, 4])

            x, y, w, h = tf.unstack(refine_boxes_pred, axis=1)
            refine_boxes_pred_angle = tf.transpose(tf.stack([x, y, w, h, angle_cls]))

            if self.cfgs.ANGLE_RANGE == 180:
                # _, _, _, _, theta = tf.unstack(boxes_pred, axis=1)
                # indx = tf.reshape(tf.where(tf.logical_and(tf.less(theta, 0), tf.greater_equal(theta, -180))), [-1, ])
                # boxes_pred = tf.gather(boxes_pred, indx)
                # scores = tf.gather(scores, indx)

                # boxes_pred = tf.py_func(coordinate_present_convert,
                #                         inp=[boxes_pred, 1],
                #                         Tout=[tf.float32])
                # boxes_pred = tf.reshape(boxes_pred, [-1, 5])

                refine_boxes_pred_angle = tf.py_func(coordinate_present_convert,
                                                     inp=[refine_boxes_pred_angle, 1],
                                                     Tout=[tf.float32])
                refine_boxes_pred_angle = tf.reshape(refine_boxes_pred_angle, [-1, 5])

            max_output_size = 4000 if 'DOTA' in self.cfgs.NET_NAME else 200
            nms_indices = nms_rotate.nms_rotate(decode_boxes=refine_boxes_pred_angle,
                                                scores=scores,
                                                iou_threshold=self.cfgs.NMS_IOU_THRESHOLD,
                                                max_output_size=100 if self.is_training else max_output_size,
                                                use_gpu=True,
                                                gpu_id=gpu_id)

            # tmp_boxes_pred = tf.reshape(tf.gather(boxes_pred, nms_indices), [-1, 5])
            tmp_refine_boxes_pred_angle = tf.reshape(tf.gather(refine_boxes_pred_angle, nms_indices), [-1, 5])
            tmp_scores = tf.reshape(tf.gather(scores, nms_indices), [-1, ])

            # return_boxes_pred.append(tmp_boxes_pred)
            return_boxes_pred_angle.append(tmp_refine_boxes_pred_angle)
            return_scores.append(tmp_scores)
            return_labels.append(tf.ones_like(tmp_scores) * (j + 1))

        # return_boxes_pred = tf.concat(return_boxes_pred, axis=0)
        return_boxes_pred_angle = tf.concat(return_boxes_pred_angle, axis=0)
        return_scores = tf.concat(return_scores, axis=0)
        return_labels = tf.concat(return_labels, axis=0)

        return return_scores, return_labels, return_boxes_pred_angle
