# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.models.detectors.single_stage_base_network import DetectionNetworkBase
from libs.models.losses.losses_rsdet import LossRSDet
from libs.utils import bbox_transform, nms_rotate
from libs.utils.coordinate_convert import backward_convert
from libs.models.samplers.rsdet.anchor_sampler_rsdet_8p import AnchorSamplerRSDet


class DetectionNetworkRSDet(DetectionNetworkBase):

    def __init__(self, cfgs, is_training):
        super(DetectionNetworkRSDet, self).__init__(cfgs, is_training)
        self.anchor_sampler_rsdet = AnchorSamplerRSDet(cfgs)
        self.losses = LossRSDet(self.cfgs)

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
                                      num_outputs=8 * self.num_anchors_per_location,
                                      kernel_size=[3, 3],
                                      stride=1,
                                      weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                      biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                      scope=scope_list[3],
                                      activation_fn=None,
                                      trainable=self.is_training,
                                      reuse=reuse_flag)

        rpn_delta_boxes = tf.reshape(rpn_delta_boxes, [-1, 8],
                                     name='rpn_{}_regression_reshape'.format(level))
        return rpn_delta_boxes

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch_h=None, gtboxes_batch_r=None, gpu_id=0):

        if self.is_training:
            gtboxes_batch_h = tf.reshape(gtboxes_batch_h, [-1, 5])
            gtboxes_batch_h = tf.cast(gtboxes_batch_h, tf.float32)

            gtboxes_batch_r = tf.reshape(gtboxes_batch_r, [-1, 9])
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
                labels, anchor_states, target_boxes = tf.py_func(
                    func=self.anchor_sampler_rsdet.anchor_target_layer,
                    inp=[gtboxes_batch_h, gtboxes_batch_r, anchors, gpu_id],
                    Tout=[tf.float32, tf.float32, tf.float32])

                if self.method == 'H':
                    self.add_anchor_img_smry(input_img_batch, anchors, anchor_states, 0)
                else:
                    self.add_anchor_img_smry(input_img_batch, anchors, anchor_states, 1)

                cls_loss = self.losses.focal_loss(labels, rpn_cls_score, anchor_states)

                reg_loss = self.losses.modulated_rotation_8p_loss(target_boxes, rpn_box_pred, anchor_states, anchors)

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

        def filter_detections(boxes, scores, is_training, gpu_id):

            if is_training:
                indices = tf.reshape(tf.where(tf.greater(scores, self.cfgs.VIS_SCORE)), [-1, ])
            else:
                indices = tf.reshape(tf.where(tf.greater(scores, self.cfgs.FILTERED_SCORE)), [-1, ])

            if self.cfgs.NMS:
                filtered_boxes = tf.gather(boxes, indices)
                filtered_scores = tf.gather(scores, indices)

                filtered_boxes = tf.py_func(func=backward_convert,
                                            inp=[filtered_boxes, False],
                                            Tout=[tf.float32])
                filtered_boxes = tf.reshape(filtered_boxes, [-1, 5])

                max_output_size = 4000 if 'DOTA' in self.cfgs.NET_NAME else 200
                nms_indices = nms_rotate.nms_rotate(decode_boxes=filtered_boxes,
                                                    scores=tf.reshape(filtered_scores, [-1, ]),
                                                    iou_threshold=self.cfgs.NMS_IOU_THRESHOLD,
                                                    max_output_size=100 if is_training else max_output_size,
                                                    use_gpu=not is_training,
                                                    gpu_id=gpu_id)

                # filter indices based on NMS
                indices = tf.gather(indices, nms_indices)

            # add indices to list of all indices
            # return indices
            return indices

        if self.cfgs.METHOD == 'H':
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            w = anchors[:, 2] - anchors[:, 0] + 1
            h = anchors[:, 3] - anchors[:, 1] + 1
            anchors = tf.transpose(tf.stack([x_c, y_c, w, h]))

        boxes_pred = bbox_transform.qbbox_transform_inv(boxes=anchors, deltas=rpn_bbox_pred)

        return_boxes_pred = []
        return_scores = []
        return_labels = []
        for j in range(0, self.cfgs.CLASS_NUM):
            indices = filter_detections(boxes_pred, rpn_cls_prob[:, j], self.is_training, gpu_id)
            tmp_boxes_pred = tf.reshape(tf.gather(boxes_pred, indices), [-1, 8])
            tmp_scores = tf.reshape(tf.gather(rpn_cls_prob[:, j], indices), [-1, ])

            return_boxes_pred.append(tmp_boxes_pred)
            return_scores.append(tmp_scores)
            return_labels.append(tf.ones_like(tmp_scores) * (j + 1))

        return_boxes_pred = tf.concat(return_boxes_pred, axis=0)
        return_scores = tf.concat(return_scores, axis=0)
        return_labels = tf.concat(return_labels, axis=0)

        return return_boxes_pred, return_scores, return_labels
