# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.models.anchor_heads.generate_anchors import GenerateAnchors
from libs.utils.show_box_in_tensor import DrawBoxTensor
from libs.models.backbones.build_backbone_p2top6 import BuildBackbone
from utils.box_ops import clip_boxes_to_img_boundaries
from libs.utils import bbox_transform
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo


class DetectionNetworkBase(object):

    def __init__(self, cfgs, is_training):

        self.cfgs = cfgs
        self.base_network_name = cfgs.NET_NAME
        self.is_training = is_training
        if cfgs.ANCHOR_MODE == 'H':
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        else:
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS) * len(cfgs.ANCHOR_ANGLES)
        self.anchor_mode = cfgs.ANCHOR_MODE
        self.losses_dict = {}
        self.drawer = DrawBoxTensor(cfgs)
        self.backbone = BuildBackbone(cfgs, is_training)
        self.pretrain_zoo = PretrainModelZoo()

    def build_backbone(self, input_img_batch):
        return self.backbone.build_backbone(input_img_batch)

    def make_anchors(self, feature_pyramid):
        with tf.variable_scope('make_anchors'):
            anchor = GenerateAnchors(self.cfgs, self.anchor_mode)
            anchor_list = anchor.generate_all_anchor(feature_pyramid)
        return anchor_list

    def rpn(self, feature_pyramid):
        with tf.variable_scope('build_rpn',
                               regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):

            fpn_cls_score = []
            fpn_box_pred = []
            for level_name in self.cfgs.LEVEL:
                if self.cfgs.SHARE_HEADS:
                    reuse_flag = None if level_name == self.cfgs.LEVEL[0] else True
                    scope_list = ['rpn_conv/3x3', 'rpn_cls_score', 'rpn_bbox_pred']
                else:
                    reuse_flag = None
                    scope_list = ['rpn_conv/3x3_%s' % level_name, 'rpn_cls_score_%s' % level_name,
                                  'rpn_bbox_pred_%s' % level_name]
                rpn_conv3x3 = slim.conv2d(
                    feature_pyramid[level_name], self.cfgs.FPN_CHANNEL, [3, 3],
                    trainable=self.is_training, weights_initializer=self.cfgs.INITIALIZER, padding="SAME",
                    activation_fn=tf.nn.relu,
                    scope=scope_list[0],
                    reuse=reuse_flag)
                rpn_cls_score = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location * 2, [1, 1], stride=1,
                                            trainable=self.is_training, weights_initializer=self.cfgs.INITIALIZER,
                                            activation_fn=None, padding="VALID",
                                            scope=scope_list[1],
                                            reuse=reuse_flag)
                rpn_box_pred = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location * 4, [1, 1], stride=1,
                                           trainable=self.is_training, weights_initializer=self.cfgs.BBOX_INITIALIZER,
                                           activation_fn=None, padding="VALID",
                                           scope=scope_list[2],
                                           reuse=reuse_flag)
                rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
                rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])

                fpn_cls_score.append(rpn_cls_score)
                fpn_box_pred.append(rpn_box_pred)

            fpn_cls_score = tf.concat(fpn_cls_score, axis=0, name='fpn_cls_score')
            fpn_box_pred = tf.concat(fpn_box_pred, axis=0, name='fpn_box_pred')
            fpn_cls_prob = slim.softmax(fpn_cls_score, scope='fpn_cls_prob')

            return fpn_box_pred, fpn_cls_score, fpn_cls_prob

    def postprocess_rpn_proposals(self, rpn_bbox_pred, rpn_cls_prob, img_shape, anchors, is_training):
        '''
        :param rpn_bbox_pred: [-1, 4]
        :param rpn_cls_prob: [-1, 2]
        :param img_shape:
        :param anchors:[-1, 4]
        :param is_training:
        :return:
        '''

        if is_training:
            pre_nms_topN = self.cfgs.RPN_TOP_K_NMS_TRAIN
            post_nms_topN = self.cfgs.RPN_MAXIMUM_PROPOSAL_TARIN
            # pre_nms_topN = self.cfgs.FPN_TOP_K_PER_LEVEL_TRAIN
            # post_nms_topN = pre_nms_topN
        else:
            pre_nms_topN = self.cfgs.RPN_TOP_K_NMS_TEST
            post_nms_topN = self.cfgs.RPN_MAXIMUM_PROPOSAL_TEST
            # pre_nms_topN = self.cfgs.FPN_TOP_K_PER_LEVEL_TEST
            # post_nms_topN = pre_nms_topN

        nms_thresh = self.cfgs.RPN_NMS_IOU_THRESHOLD

        cls_prob = rpn_cls_prob[:, 1]

        # 1. decode boxes
        decode_boxes = bbox_transform.bbox_transform_inv(boxes=anchors, deltas=rpn_bbox_pred,
                                                         scale_factors=self.cfgs.ANCHOR_SCALE_FACTORS)

        # 2. clip to img boundaries
        decode_boxes = clip_boxes_to_img_boundaries(decode_boxes=decode_boxes,
                                                    img_shape=img_shape)

        # 3. get top N to NMS
        if pre_nms_topN > 0:
            pre_nms_topN = tf.minimum(pre_nms_topN, tf.shape(decode_boxes)[0], name='avoid_unenough_boxes')
            cls_prob, top_k_indices = tf.nn.top_k(cls_prob, k=pre_nms_topN)
            decode_boxes = tf.gather(decode_boxes, top_k_indices)

        # 4. NMS
        keep = tf.image.non_max_suppression(
            boxes=decode_boxes,
            scores=cls_prob,
            max_output_size=post_nms_topN,
            iou_threshold=nms_thresh)

        final_boxes = tf.gather(decode_boxes, keep)
        final_probs = tf.gather(cls_prob, keep)

        return final_boxes, final_probs

    def add_anchor_img_smry(self, img, anchors, labels, method):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = self.drawer.only_draw_boxes(img_batch=img,
                                                 boxes=positive_anchor,
                                                 method=method)
        neg_in_img = self.drawer.only_draw_boxes(img_batch=img,
                                                 boxes=negative_anchor,
                                                 method=method)

        tf.summary.image('positive_anchor', pos_in_img)
        tf.summary.image('negative_anchors', neg_in_img)

    def add_roi_batch_img_smry(self, img, rois, labels, method):
        positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])

        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)

        pos_in_img = self.drawer.only_draw_boxes(img_batch=img,
                                                 boxes=pos_roi,
                                                 method=method)
        neg_in_img = self.drawer.only_draw_boxes(img_batch=img,
                                                 boxes=neg_roi,
                                                 method=method)

        tf.summary.image('pos_rois', pos_in_img)
        tf.summary.image('neg_rois', neg_in_img)

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(self.cfgs.TRAINED_CKPT, self.cfgs.VERSION))
        if checkpoint_path is not None:
            if self.cfgs.RESTORE_FROM_RPN:
                print('___restore from rpn___')
                model_variables = slim.get_model_variables()
                restore_variables = [var for var in model_variables if not var.name.startswith('FastRCNN_Head')] + \
                                    [slim.get_or_create_global_step()]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)
        else:
            if self.cfgs.NET_NAME in self.pretrain_zoo.pth_zoo:
                return None, None
            checkpoint_path = self.cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()

            # for var in model_variables:
            #     print(var.name)
            # print(20*"__++__++__")

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                Fast-RCNN/MobilenetV2/** -- > MobilenetV2 **
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith('Fast-RCNN/'+self.base_network_name):  # +'/block4'
                    var_name_in_ckpt = name_in_ckpt_fastrcnn_head(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
                else:
                    if var.name.startswith(self.base_network_name):
                        var_name_in_ckpt = name_in_ckpt_rpn(var)
                        nameInCkpt_Var_dict[var_name_in_ckpt] = var
                    else:
                        continue
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"___")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path

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
                                                                      tf.zeros(shape=(1, 5 * (self.cfgs.CLASS_NUM + 1)),
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


