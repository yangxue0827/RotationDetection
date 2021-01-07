# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

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

                if self.cfgs.REG_LOSS_MODE == 0:
                    reg_loss = self.losses.iou_smooth_l1_loss_log(target_delta, rpn_box_pred, anchor_states,
                                                                  target_boxes, anchors)
                elif self.cfgs.REG_LOSS_MODE == 1:
                    reg_loss = self.losses.iou_smooth_l1_loss_exp(target_delta, rpn_box_pred, anchor_states,
                                                                  target_boxes, anchors, alpha=self.cfgs.ALPHA,
                                                                  beta=self.cfgs.BETA)
                else:
                    reg_loss = self.losses.smooth_l1_loss(target_delta, rpn_box_pred, anchor_states)

                self.losses_dict['cls_loss'] = cls_loss * self.cfgs.CLS_WEIGHT
                self.losses_dict['reg_loss'] = reg_loss * self.cfgs.REG_WEIGHT

        return rpn_box_pred, rpn_cls_prob
