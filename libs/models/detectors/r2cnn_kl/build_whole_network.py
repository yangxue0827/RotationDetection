# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.models.detectors.two_stage_base_network import DetectionNetworkBase
from libs.models.losses.losses_kl import LossKL
from libs.utils import bbox_transform, nms_rotate
from libs.models.samplers.r2cnn.anchor_sampler_r2cnn import AnchorSamplerR2CNN
from libs.models.samplers.r2cnn.proposal_sampler_r2cnn import ProposalSamplerR2CNN
from libs.models.roi_extractors.roi_extractors import RoIExtractor
from libs.models.box_heads.box_head_base import BoxHead


class DetectionNetworkR2CNNKL(DetectionNetworkBase):

    def __init__(self, cfgs, is_training):
        super(DetectionNetworkR2CNNKL, self).__init__(cfgs, is_training)
        self.anchor_sampler_r2cnn = AnchorSamplerR2CNN(cfgs)
        self.proposal_sampler_r2cnn = ProposalSamplerR2CNN(cfgs)
        self.losses = LossKL(cfgs)
        self.roi_extractor = RoIExtractor(cfgs)
        self.box_head = BoxHead(cfgs)

    def assign_levels(self, all_rois, labels=None, bbox_targets=None):
        '''
        :param all_rois:
        :param labels:
        :param bbox_targets:
        :return:
        '''
        with tf.name_scope('assign_levels'):
            # all_rois = tf.Print(all_rois, [tf.shape(all_rois)], summarize=10, message='ALL_ROIS_SHAPE*****')
            xmin, ymin, xmax, ymax = tf.unstack(all_rois, axis=1)

            h = tf.maximum(0., ymax - ymin)
            w = tf.maximum(0., xmax - xmin)

            levels = tf.floor(4. + tf.log(tf.sqrt(w * h + 1e-8) / 224.0) / tf.log(2.))  # 4 + log_2(***)
            # use floor instead of round

            min_level = int(self.cfgs.LEVEL[0][-1])
            max_level = min(5, int(self.cfgs.LEVEL[-1][-1]))
            levels = tf.maximum(levels, tf.ones_like(levels) * min_level)  # level minimum is 2
            levels = tf.minimum(levels, tf.ones_like(levels) * max_level)  # level maximum is 5

            levels = tf.stop_gradient(tf.reshape(levels, [-1]))

            def get_rois(levels, level_i, rois, labels, bbox_targets):

                level_i_indices = tf.reshape(tf.where(tf.equal(levels, level_i)), [-1])

                tf.summary.scalar('LEVEL/LEVEL_%d_rois_NUM' % level_i, tf.shape(level_i_indices)[0])
                level_i_rois = tf.gather(rois, level_i_indices)

                if self.is_training:
                    if not self.cfgs.CUDA8:
                        # Note: for > cuda 9
                        level_i_rois = tf.stop_gradient(level_i_rois)
                        level_i_labels = tf.gather(labels, level_i_indices)

                        level_i_targets = tf.gather(bbox_targets, level_i_indices)
                    else:

                        # Note: for cuda 8
                        level_i_rois = tf.stop_gradient(tf.concat([level_i_rois, [[0, 0, 0., 0.]]], axis=0))
                        # to avoid the num of level i rois is 0.0, which will broken the BP in tf

                        level_i_labels = tf.gather(labels, level_i_indices)
                        level_i_labels = tf.stop_gradient(tf.concat([level_i_labels, [0]], axis=0))

                        level_i_targets = tf.gather(bbox_targets, level_i_indices)
                        level_i_targets = tf.stop_gradient(tf.concat([level_i_targets,
                                                                      tf.zeros(shape=(1, 5),
                                                                               dtype=tf.float32)], axis=0))

                    return level_i_rois, level_i_labels, level_i_targets
                else:
                    if self.cfgs.CUDA8:
                        # Note: for cuda 8
                        level_i_rois = tf.concat([level_i_rois, [[0, 0, 0., 0.]]], axis=0)
                    return level_i_rois, None, None

            rois_list = []
            labels_list = []
            targets_list = []
            for i in range(min_level, max_level + 1):
                P_i_rois, P_i_labels, P_i_targets = get_rois(levels, level_i=i, rois=all_rois,
                                                             labels=labels,
                                                             bbox_targets=bbox_targets)
                rois_list.append(P_i_rois)
                labels_list.append(P_i_labels)
                targets_list.append(P_i_targets)

            if self.is_training:
                all_labels = tf.concat(labels_list, axis=0)
                all_targets = tf.concat(targets_list, axis=0)
                return rois_list, all_labels, all_targets
            else:
                return rois_list  # [P2_rois, P3_rois, P4_rois, P5_rois] Note: P6 do not assign rois

    def build_loss(self, rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels,
                   bbox_pred, bbox_targets, rois, target_gt, cls_score, labels):

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
                reg_loss = self.losses.KL_divergence_loss_two_stage(bbox_pred=bbox_pred,
                                                                    rois=rois,
                                                                    target_gt=target_gt,
                                                                    label=labels,
                                                                    num_classes=self.cfgs.CLASS_NUM + 1,
                                                                    tau=self.cfgs.KL_TAU,
                                                                    func=self.cfgs.KL_FUNC)

                cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=cls_score,
                    labels=labels))

                self.losses_dict['fast_cls_loss'] = cls_loss * self.cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                self.losses_dict['fast_reg_loss'] = reg_loss * self.cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch_h=None, gtboxes_batch_r=None, gpu_id=0):

        if self.is_training:
            gtboxes_batch_h = tf.reshape(gtboxes_batch_h, [-1, 5])
            gtboxes_batch_h = tf.cast(gtboxes_batch_h, tf.float32)

            gtboxes_batch_r = tf.reshape(gtboxes_batch_r, [-1, 6])
            gtboxes_batch_r = tf.cast(gtboxes_batch_r, tf.float32)

        img_shape = tf.shape(input_img_batch)

        # 1. build backbone
        feature_pyramid = self.build_backbone(input_img_batch)

        # 2. build rpn
        fpn_box_pred, fpn_cls_score, fpn_cls_prob = self.rpn(feature_pyramid)

        # 3. generate anchors
        anchor_list = self.make_anchors(feature_pyramid)
        anchors = tf.concat(anchor_list, axis=0)

        # 4. postprocess rpn proposals. such as: decode, clip, NMS
        with tf.variable_scope('postprocess_FPN'):
            rois, roi_scores = self.postprocess_rpn_proposals(rpn_bbox_pred=fpn_box_pred,
                                                              rpn_cls_prob=fpn_cls_prob,
                                                              img_shape=img_shape,
                                                              anchors=anchors,
                                                              is_training=self.is_training)

        # 5. sample minibatch
        if self.is_training:
            with tf.variable_scope('sample_anchors_minibatch'):
                fpn_labels, fpn_bbox_targets = \
                    tf.py_func(
                        self.anchor_sampler_r2cnn.anchor_target_layer,
                        [gtboxes_batch_h, img_shape, anchors],
                        [tf.float32, tf.float32])
                fpn_bbox_targets = tf.reshape(fpn_bbox_targets, [-1, 4])
                fpn_labels = tf.to_int32(fpn_labels, name="to_int32")
                fpn_labels = tf.reshape(fpn_labels, [-1])
                self.add_anchor_img_smry(input_img_batch, anchors, fpn_labels, method=0)

            fpn_cls_category = tf.argmax(fpn_cls_prob, axis=1)
            kept_rpppn = tf.reshape(tf.where(tf.not_equal(fpn_labels, -1)), [-1])
            fpn_cls_category = tf.gather(fpn_cls_category, kept_rpppn)
            acc = tf.reduce_mean(tf.to_float(tf.equal(fpn_cls_category,
                                                      tf.to_int64(tf.gather(fpn_labels, kept_rpppn)))))
            tf.summary.scalar('ACC/fpn_accuracy', acc)

            with tf.control_dependencies([fpn_labels]):

                with tf.variable_scope('sample_RCNN_minibatch'):
                    rois, labels, _, bbox_targets, _, target_gt = \
                        tf.py_func(self.proposal_sampler_r2cnn.proposal_target_layer,
                                   [rois, gtboxes_batch_h, gtboxes_batch_r],
                                   [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
                    rois = tf.reshape(rois, [-1, 4])
                    labels = tf.to_int32(labels)
                    labels = tf.reshape(labels, [-1])
                    bbox_targets = tf.reshape(bbox_targets, [-1, 5 * (self.cfgs.CLASS_NUM + 1)])
                    target_gt = tf.reshape(target_gt, [-1, 5])
                    self.add_roi_batch_img_smry(input_img_batch, rois, labels, method=0)

        # 6. assign level
        if self.is_training:
            rois_list, labels, target_gt = self.assign_levels(all_rois=rois,
                                                              labels=labels,
                                                              bbox_targets=target_gt)
        else:
            rois_list = self.assign_levels(all_rois=rois)

        # 7. build Fast-RCNN, include roi align/pooling, box head
        bbox_pred, cls_score = self.box_head.fpn_fc_head(self.roi_extractor, rois_list, feature_pyramid,
                                                         img_shape, self.is_training)
        rois = tf.concat(rois_list, axis=0, name='concat_rois')
        cls_prob = slim.softmax(cls_score, 'cls_prob')

        if self.is_training:
            cls_category = tf.argmax(cls_prob, axis=1)
            fast_acc = tf.reduce_mean(tf.to_float(tf.equal(cls_category, tf.to_int64(labels))))
            tf.summary.scalar('ACC/fast_acc', fast_acc)

        #  8. build loss
        if self.is_training:
            self.build_loss(rpn_box_pred=fpn_box_pred,
                            rpn_bbox_targets=fpn_bbox_targets,
                            rpn_cls_score=fpn_cls_score,
                            rpn_labels=fpn_labels,
                            bbox_pred=bbox_pred,
                            bbox_targets=bbox_targets,
                            rois=rois,
                            target_gt=target_gt,
                            cls_score=cls_score,
                            labels=labels)

        # 9. postprocess_fastrcnn
        final_bbox, final_scores, final_category = self.postprocess_fastrcnn(rois=rois,
                                                                             bbox_ppred=bbox_pred,
                                                                             scores=cls_prob,
                                                                             gpu_id=gpu_id)
        if self.is_training:
            return final_bbox, final_scores, final_category, self.losses_dict
        else:
            return final_bbox, final_scores, final_category

    def postprocess_fastrcnn(self, rois, bbox_ppred, scores, gpu_id):
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

                if self.is_training:
                    indices = tf.reshape(tf.where(tf.greater(tmp_score, self.cfgs.VIS_SCORE)), [-1, ])
                else:
                    indices = tf.reshape(tf.where(tf.greater(tmp_score, self.cfgs.FILTERED_SCORE)), [-1, ])

                rois_ = tf.gather(rois, indices)
                tmp_score = tf.gather(tmp_score, indices)
                tmp_encoded_box = tf.gather(tmp_encoded_box, indices)

                # 1. decode boxes in each class
                tmp_decoded_boxes = bbox_transform.rbbox_transform_inv(boxes=rois_, deltas=tmp_encoded_box,
                                                                       scale_factors=self.cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                # tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                #                                                              img_shape=img_shape)

                # 3. NMS
                max_output_size = 4000 if 'DOTA' in self.cfgs.NET_NAME else 200
                keep = nms_rotate.nms_rotate(decode_boxes=tmp_decoded_boxes,
                                             scores=tmp_score,
                                             iou_threshold=self.cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                             max_output_size=100 if self.is_training else max_output_size,
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

            return final_boxes, final_scores, final_category
