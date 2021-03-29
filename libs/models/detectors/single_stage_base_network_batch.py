# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.models.anchor_heads.generate_anchors import GenerateAnchors
from libs.utils.show_box_in_tensor import DrawBoxTensor
from libs.models.backbones.build_backbone_p3top7 import BuildBackbone
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo


class DetectionNetworkBase(object):

    def __init__(self, cfgs, is_training):

        self.cfgs = cfgs
        self.base_network_name = cfgs.NET_NAME
        self.is_training = is_training
        self.batch_size = cfgs.BATCH_SIZE if is_training else 1
        if cfgs.METHOD == 'H':
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        else:
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS) * len(cfgs.ANCHOR_ANGLES)
        self.method = cfgs.METHOD
        self.losses_dict = {}
        self.drawer = DrawBoxTensor(cfgs)
        self.backbone = BuildBackbone(cfgs, is_training)
        self.pretrain_zoo = PretrainModelZoo()

    def build_backbone(self, input_img_batch):
        return self.backbone.build_backbone(input_img_batch)

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
                                     num_outputs=self.cfgs.CLASS_NUM * self.num_anchors_per_location,
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
        rpn_box_probs = tf.sigmoid(rpn_box_scores, name='rpn_{}_classification_sigmoid'.format(level))

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

        rpn_delta_boxes = slim.conv2d(rpn_conv2d_3x3,
                                      num_outputs=5 * self.num_anchors_per_location,
                                      kernel_size=[3, 3],
                                      stride=1,
                                      weights_initializer=self.cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                      biases_initializer=self.cfgs.SUBNETS_BIAS_INITIALIZER,
                                      scope=scope_list[3],
                                      activation_fn=None,
                                      trainable=self.is_training,
                                      reuse=reuse_flag)

        rpn_delta_boxes = tf.reshape(rpn_delta_boxes, [self.batch_size, -1, 5],
                                     name='rpn_{}_regression_reshape'.format(level))
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

            return rpn_delta_boxes_list, rpn_scores_list, rpn_probs_list

    def make_anchors(self, feature_pyramid, use_tf=True):
        with tf.variable_scope('make_anchors'):
            anchor = GenerateAnchors(self.cfgs, self.method)
            if use_tf and self.method == 'H':
                anchor_list = anchor.generate_all_anchor_tf(feature_pyramid)
            else:
                anchor_list = anchor.generate_all_anchor(feature_pyramid)
        return anchor_list

    def add_anchor_img_smry(self, img, anchors, labels, method):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        # negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        # negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = self.drawer.only_draw_boxes(img_batch=img,
                                                 boxes=positive_anchor,
                                                 method=method)
        # neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
        #                                                 boxes=negative_anchor)

        tf.summary.image('positive_anchor', pos_in_img)
        # tf.summary.image('negative_anchors', neg_in_img)

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



