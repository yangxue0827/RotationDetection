# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from libs.models.detectors.single_stage_base_network import DetectionNetworkBase
from libs.models.losses.losses_gwd import LossGWD
from libs.utils import bbox_transform, nms_rotate
from libs.models.anchor_heads.generate_anchors import GenerateAnchors
from libs.models.samplers.gwd.anchor_sampler_gwd import AnchorSamplerGWD


class DetectionNetworkGWD(DetectionNetworkBase):

    def __init__(self, cfgs, is_training):
        super(DetectionNetworkGWD, self).__init__(cfgs, is_training)
        self.anchor_sampler_gwd = AnchorSamplerGWD(cfgs)
        self.losses = LossGWD(self.cfgs)

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
                labels, target_delta, anchor_states, target_boxes = tf.py_func(func=self.anchor_sampler_gwd.anchor_target_layer,
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
                elif self.cfgs.REG_LOSS_MODE == 2:
                    reg_loss = self.losses.wasserstein_distance_loss(rpn_box_pred, anchor_states,
                                                                     target_boxes, anchors, tau=self.cfgs.GWD_TAU,
                                                                     func=self.cfgs.GWD_FUNC)
                else:
                    reg_loss = self.losses.smooth_l1_loss(target_delta, rpn_box_pred, anchor_states)

                self.losses_dict['cls_loss'] = cls_loss * self.cfgs.CLS_WEIGHT
                self.losses_dict['reg_loss'] = reg_loss * self.cfgs.REG_WEIGHT

        # 5. postprocess
        # with tf.variable_scope('postprocess_detctions'):
        #     boxes, scores, category = self.postprocess_detctions(rpn_bbox_pred=rpn_box_pred,
        #                                                          rpn_cls_prob=rpn_cls_prob,
        #                                                          feature_pyramid=feature_pyramid)
        #     boxes = tf.stop_gradient(boxes)
        #     scores = tf.stop_gradient(scores)
        #     category = tf.stop_gradient(category)
        #
        # if self.is_training:
        #     return boxes, scores, category, self.losses_dict
        # else:
        #     return boxes, scores, category
        return rpn_box_pred, rpn_cls_prob

    def postprocess_detctions(self, rpn_bbox_pred, rpn_cls_prob, feature_pyramid):
        anchor = GenerateAnchors(self.cfgs, self.method)
        # anchor_list = anchor.generate_all_anchor(feature_pyramid)
        anchor_list = anchor.generate_all_anchor_tf(feature_pyramid)
        anchors = tf.concat(anchor_list, axis=0)

        return_boxes_pred = []
        return_scores = []
        return_labels = []
        for j in range(0, self.cfgs.CLASS_NUM):
            scores = rpn_cls_prob[:, j]
            indices = tf.reshape(tf.where(tf.greater(scores, self.cfgs.FILTERED_SCORE)), [-1, ])

            anchors_ = tf.gather(anchors, indices)
            rpn_bbox_pred_ = tf.gather(rpn_bbox_pred, indices)
            scores = tf.gather(scores, indices)

            x_c = (anchors_[:, 2] + anchors_[:, 0]) / 2
            y_c = (anchors_[:, 3] + anchors_[:, 1]) / 2
            h = anchors_[:, 2] - anchors_[:, 0] + 1
            w = anchors_[:, 3] - anchors_[:, 1] + 1
            theta = -90 * tf.ones_like(x_c)
            anchors_ = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

            boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors_, deltas=rpn_bbox_pred_)

            # nms_indices = nms_rotate.nms_rotate(decode_boxes=boxes_pred,
            #                                     scores=scores,
            #                                     iou_threshold=self.cfgs.NMS_IOU_THRESHOLD,
            #                                     max_output_size=100 if self.is_training else 1000,
            #                                     use_gpu=False)
            #
            # tmp_boxes_pred = tf.reshape(tf.gather(boxes_pred, nms_indices), [-1, 5])
            # tmp_scores = tf.reshape(tf.gather(scores, nms_indices), [-1, ])

            tmp_boxes_pred = tf.reshape(boxes_pred, [-1, 5])
            tmp_scores = tf.reshape(scores, [-1, ])

            return_boxes_pred.append(tmp_boxes_pred)
            return_scores.append(tmp_scores)
            return_labels.append(tf.ones_like(scores) * (j + 1))

        return_boxes_pred = tf.concat(return_boxes_pred, axis=0)
        return_scores = tf.concat(return_scores, axis=0)
        return_labels = tf.concat(return_labels, axis=0)

        return return_boxes_pred, return_scores, return_labels
