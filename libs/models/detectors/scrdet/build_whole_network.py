# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.models.detectors.two_stage_base_network import DetectionNetworkBase
from libs.models.losses.losses import Loss
from libs.utils import bbox_transform, nms_rotate
from libs.models.anchor_heads import generate_h_anchors, anchor_utils
from libs.models.samplers.r2cnn.anchor_sampler_r2cnn import AnchorSamplerR2CNN
from libs.models.samplers.r2cnn.proposal_sampler_r2cnn import ProposalSamplerR2CNN
from libs.models.roi_extractors.roi_extractors import RoIExtractor
from libs.models.box_heads.box_head_base import BoxHead
from utils.box_ops import clip_boxes_to_img_boundaries


class DetectionNetworkSCRDet(DetectionNetworkBase):

    def __init__(self, cfgs, is_training):
        super(DetectionNetworkSCRDet, self).__init__(cfgs, is_training)
        self.proposal_sampler_r2cnn = ProposalSamplerR2CNN(cfgs)
        self.anchor_sampler_r2cnn = AnchorSamplerR2CNN(cfgs)
        self.losses = Loss(cfgs)
        self.roi_extractor = RoIExtractor(cfgs)
        self.box_head = BoxHead(cfgs)

    def rpn(self, inputs):
        rpn_conv3x3 = slim.conv2d(inputs, 512, [3, 3],
                                  trainable=self.is_training,
                                  weights_initializer=self.cfgs.INITIALIZER,
                                  activation_fn=tf.nn.relu,
                                  scope='rpn_conv/3x3')
        rpn_cls_score = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location * 2, [1, 1], stride=1,
                                    trainable=self.is_training, weights_initializer=self.cfgs.INITIALIZER,
                                    activation_fn=None,
                                    scope='rpn_cls_score')
        rpn_box_pred = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location * 4, [1, 1], stride=1,
                                   trainable=self.is_training, weights_initializer=self.cfgs.BBOX_INITIALIZER,
                                   activation_fn=None,
                                   scope='rpn_bbox_pred')
        rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob')

        return rpn_box_pred, rpn_cls_score, rpn_cls_prob

    def make_anchors(self, feature_to_cropped):
        featuremap_height, featuremap_width = tf.shape(feature_to_cropped)[1], tf.shape(feature_to_cropped)[2]
        featuremap_height = tf.cast(featuremap_height, tf.float32)
        featuremap_width = tf.cast(featuremap_width, tf.float32)

        anchors = anchor_utils.make_anchors(base_anchor_size=self.cfgs.BASE_ANCHOR_SIZE_LIST,
                                            anchor_scales=self.cfgs.ANCHOR_SCALES, anchor_ratios=self.cfgs.ANCHOR_RATIOS,
                                            featuremap_height=featuremap_height,
                                            featuremap_width=featuremap_width,
                                            stride=self.cfgs.ANCHOR_STRIDE,
                                            name="make_anchors_forRPN")
        return anchors

    def build_loss(self, rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels,
                   bbox_pred_h, bbox_targets_h, cls_score_h, bbox_pred_r, bbox_targets_r, rois, target_gt_r,
                   cls_score_r, labels, mask_gt, pa_mask_pred):
        '''
        :param rpn_box_pred: [-1, 4]
        :param rpn_bbox_targets: [-1, 4]
        :param rpn_cls_score: [-1]
        :param rpn_labels: [-1]
        :param bbox_pred_h: [-1, 4*(cls_num+1)]
        :param bbox_targets_h: [-1, 4*(cls_num+1)]
        :param cls_score_h: [-1, cls_num+1]
        :param bbox_pred_r: [-1, 5*(cls_num+1)]
        :param bbox_targets_r: [-1, 5*(cls_num+1)]
        :param cls_score_r: [-1, cls_num+1]
        :param labels: [-1]
        :return:
        '''

        with tf.variable_scope('build_loss'):

            with tf.variable_scope('rpn_loss'):

                rpn_reg_loss = self.losses.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                                                              bbox_targets=rpn_bbox_targets,
                                                              label=rpn_labels,
                                                              sigma=self.cfgs.RPN_SIGMA)
                rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
                rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
                rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), [-1])
                rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                                                             labels=rpn_labels))

                self.losses_dict['rpn_cls_loss'] = rpn_cls_loss * self.cfgs.RPN_CLASSIFICATION_LOSS_WEIGHT
                self.losses_dict['rpn_reg_loss'] = rpn_reg_loss * self.cfgs.RPN_LOCATION_LOSS_WEIGHT

            with tf.variable_scope('FastRCNN_loss'):

                reg_loss_h = self.losses.smooth_l1_loss_rcnn_h(bbox_pred=bbox_pred_h,
                                                               bbox_targets=bbox_targets_h,
                                                               label=labels,
                                                               num_classes=self.cfgs.CLASS_NUM + 1,
                                                               sigma=self.cfgs.FASTRCNN_SIGMA)
                if self.cfgs.USE_IOU_FACTOR:
                    reg_loss_r = self.losses.iou_smooth_l1_loss_rcnn_r(bbox_pred=bbox_pred_r,
                                                                       bbox_targets=bbox_targets_r,
                                                                       label=labels,
                                                                       rois=rois, target_gt_r=target_gt_r,
                                                                       num_classes=self.cfgs.CLASS_NUM + 1,
                                                                       sigma=self.cfgs.FASTRCNN_SIGMA)
                else:
                    reg_loss_r = self.losses.smooth_l1_loss_rcnn_r(bbox_pred=bbox_pred_r,
                                                                   bbox_targets=bbox_targets_r,
                                                                   label=labels,
                                                                   num_classes=self.cfgs.CLASS_NUM + 1,
                                                                   sigma=self.cfgs.FASTRCNN_SIGMA)

                cls_loss_h = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=cls_score_h,
                    labels=labels))  # beacause already sample before
                cls_loss_r = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=cls_score_r,
                    labels=labels))

                self.losses_dict['fast_cls_loss_h'] = cls_loss_h * self.cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                self.losses_dict['fast_reg_loss_h'] = reg_loss_h * self.cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT
                self.losses_dict['fast_cls_loss_r'] = cls_loss_r * self.cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                self.losses_dict['fast_reg_loss_r'] = reg_loss_r * self.cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT

            with tf.variable_scope('build_attention_loss',
                                   regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):
                attention_loss = self.losses.build_attention_loss(mask_gt, pa_mask_pred)
                self.losses_dict['attention_loss'] = attention_loss

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch_h=None, gtboxes_batch_r=None,
                                      mask_batch=None, gpu_id=0):

        if self.is_training:
            gtboxes_batch_h = tf.reshape(gtboxes_batch_h, [-1, 5])
            gtboxes_batch_h = tf.cast(gtboxes_batch_h, tf.float32)

            gtboxes_batch_r = tf.reshape(gtboxes_batch_r, [-1, 6])
            gtboxes_batch_r = tf.cast(gtboxes_batch_r, tf.float32)

        img_shape = tf.shape(input_img_batch)

        # 1. build backbone
        feature, pa_mask = self.build_backbone(input_img_batch)

        # 2. build rpn
        rpn_box_pred, rpn_cls_score, rpn_cls_prob = self.rpn(feature)
        rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
        rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
        rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob')

        # 3. generate anchors
        anchors = self.make_anchors(feature)

        # 4. postprocess rpn proposals. such as: decode, clip, NMS
        with tf.variable_scope('postprocess_RPN'):
            rois, roi_scores = self.postprocess_rpn_proposals(rpn_bbox_pred=rpn_box_pred,
                                                              rpn_cls_prob=rpn_cls_prob,
                                                              img_shape=img_shape,
                                                              anchors=anchors,
                                                              is_training=self.is_training)

        # 5. sample minibatch
        if self.is_training:
            with tf.variable_scope('sample_anchors_minibatch'):
                rpn_labels, rpn_bbox_targets = \
                    tf.py_func(
                        self.anchor_sampler_r2cnn.anchor_target_layer,
                        [gtboxes_batch_h, img_shape, anchors],
                        [tf.float32, tf.float32])
                rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1, 4])
                rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
                rpn_labels = tf.reshape(rpn_labels, [-1])
                self.add_anchor_img_smry(input_img_batch, anchors, rpn_labels, method=0)

            rpn_cls_category = tf.argmax(rpn_cls_prob, axis=1)
            kept_rpppn = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
            rpn_cls_category = tf.gather(rpn_cls_category, kept_rpppn)
            acc = tf.reduce_mean(tf.to_float(tf.equal(rpn_cls_category,
                                                      tf.to_int64(tf.gather(rpn_labels, kept_rpppn)))))
            tf.summary.scalar('ACC/fpn_accuracy', acc)

            with tf.control_dependencies([rpn_labels]):

                with tf.variable_scope('sample_RCNN_minibatch'):
                    rois, labels, bbox_targets_h, bbox_targets_r, target_gt_h, target_gt_r = \
                        tf.py_func(self.proposal_sampler_r2cnn.proposal_target_layer,
                                   [rois, gtboxes_batch_h, gtboxes_batch_r],
                                   [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
                    rois = tf.reshape(rois, [-1, 4])
                    labels = tf.to_int32(labels)
                    labels = tf.reshape(labels, [-1])
                    bbox_targets_h = tf.reshape(bbox_targets_h, [-1, 4 * (self.cfgs.CLASS_NUM + 1)])
                    bbox_targets_r = tf.reshape(bbox_targets_r, [-1, 5 * (self.cfgs.CLASS_NUM + 1)])
                    self.add_roi_batch_img_smry(input_img_batch, rois, labels, method=0)

        # 6. build Fast-RCNN, include roi align/pooling, box head
        bbox_pred_h, cls_score_h, bbox_pred_r, cls_score_r = self.box_head.fc_head(self.roi_extractor, rois, feature,
                                                                                   img_shape, self.is_training, mode=0)
        cls_prob_h = slim.softmax(cls_score_h, 'cls_prob_h')
        cls_prob_r = slim.softmax(cls_score_r, 'cls_prob_r')

        if self.is_training:
            cls_category_h = tf.argmax(cls_prob_h, axis=1)
            fast_acc_h = tf.reduce_mean(tf.to_float(tf.equal(cls_category_h, tf.to_int64(labels))))
            tf.summary.scalar('ACC/fast_acc_h', fast_acc_h)

            cls_category_r = tf.argmax(cls_prob_r, axis=1)
            fast_acc_r = tf.reduce_mean(tf.to_float(tf.equal(cls_category_r, tf.to_int64(labels))))
            tf.summary.scalar('ACC/fast_acc_r', fast_acc_r)

        #  8. build loss
        if self.is_training:
            self.build_loss(rpn_box_pred=rpn_box_pred,
                            rpn_bbox_targets=rpn_bbox_targets,
                            rpn_cls_score=rpn_cls_score,
                            rpn_labels=rpn_labels,
                            bbox_pred_h=bbox_pred_h,
                            bbox_targets_h=bbox_targets_h,
                            cls_score_h=cls_score_h,
                            bbox_pred_r=bbox_pred_r,
                            bbox_targets_r=bbox_targets_r,
                            rois=rois,
                            target_gt_r=target_gt_r,
                            cls_score_r=cls_score_r,
                            labels=labels,
                            mask_gt=mask_batch,
                            pa_mask_pred=pa_mask)

        # 9. postprocess_fastrcnn
        final_boxes_h, final_scores_h, final_category_h = self.postprocess_fastrcnn_h(rois=rois,
                                                                                      bbox_ppred=bbox_pred_h,
                                                                                      scores=cls_prob_h,
                                                                                      img_shape=img_shape)
        final_boxes_r, final_scores_r, final_category_r = self.postprocess_fastrcnn_r(rois=rois,
                                                                                      bbox_ppred=bbox_pred_r,
                                                                                      scores=cls_prob_r,
                                                                                      gpu_id=gpu_id)
        if self.is_training:
            return final_boxes_h, final_scores_h, final_category_h, \
                   final_boxes_r, final_scores_r, final_category_r, self.losses_dict
        else:
            return final_boxes_h, final_scores_h, final_category_h, \
                   final_boxes_r, final_scores_r, final_category_r,

    def postprocess_fastrcnn_r(self, rois, bbox_ppred, scores, gpu_id):
        '''
        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 5]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, self.cfgs.CLASS_NUM + 1, 5])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []

            x_c = (rois[:, 2] + rois[:, 0]) / 2
            y_c = (rois[:, 3] + rois[:, 1]) / 2
            h = rois[:, 2] - rois[:, 0] + 1
            w = rois[:, 3] - rois[:, 1] + 1
            theta = -90 * tf.ones_like(x_c)
            rois = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
            for i in range(1, self.cfgs.CLASS_NUM + 1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]

                tmp_decoded_boxes = bbox_transform.rbbox_transform_inv(boxes=rois, deltas=tmp_encoded_box,
                                                                       scale_factors=self.cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                # tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                #                                                              img_shape=img_shape)

                # 3. NMS
                if self.cfgs.SOFT_NMS:
                    print("Using Soft NMS.......")
                    raise NotImplementedError("soft NMS for rotate has not implemented")

                else:
                    keep = nms_rotate.nms_rotate(decode_boxes=tmp_decoded_boxes,
                                                 scores=tmp_score,
                                                 iou_threshold=self.cfgs.FAST_RCNN_R_NMS_IOU_THRESHOLD,
                                                 max_output_size=self.cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                                 use_gpu=self.cfgs.ROTATE_NMS_USE_GPU,
                                                 gpu_id=gpu_id)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            if self.is_training:
                '''
                in training. We should show the detecitons in the tensorboard. So we add this.
                '''
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, self.cfgs.VIS_SCORE)), [-1])
            else:
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, self.cfgs.FILTERED_SCORE)), [-1])
            final_boxes = tf.gather(final_boxes, kept_indices)
            final_scores = tf.gather(final_scores, kept_indices)
            final_category = tf.gather(final_category, kept_indices)

            return final_boxes, final_scores, final_category

    def postprocess_fastrcnn_h(self, rois, bbox_ppred, scores, img_shape):

        '''
        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn_h'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, self.cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, self.cfgs.CLASS_NUM + 1):
                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = bbox_transform.bbox_transform_inv(boxes=rois, deltas=tmp_encoded_box,
                                                                      scale_factors=self.cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                tmp_decoded_boxes = clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                 img_shape=img_shape)

                # 3. NMS
                max_output_size = 4000 if 'DOTA' in self.cfgs.NET_NAME else 200
                keep = tf.image.non_max_suppression(
                    boxes=tmp_decoded_boxes,
                    scores=tmp_score,
                    max_output_size=100 if self.is_training else max_output_size,
                    iou_threshold=self.cfgs.FAST_RCNN_H_NMS_IOU_THRESHOLD)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            # if self.is_training:
            '''
            in training. We should show the detecitons in the tensorboard. So we add this.
            '''
            if self.is_training:
                '''
                in training. We should show the detecitons in the tensorboard. So we add this.
                '''
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, self.cfgs.VIS_SCORE)), [-1])
            else:
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, self.cfgs.FILTERED_SCORE)), [-1])
            final_boxes = tf.gather(final_boxes, kept_indices)
            final_scores = tf.gather(final_scores, kept_indices)
            final_category = tf.gather(final_category, kept_indices)

            return final_boxes, final_scores, final_category
