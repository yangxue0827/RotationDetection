# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import tensorflow.contrib.slim as slim
import tensorflow as tf

# from utils.tools import add_heatmap


class NeckSCRDet(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def fusion_two_layer(self, feat1, feat2, scope, is_training):

        with tf.variable_scope(scope):

            h, w = tf.shape(feat1)[1], tf.shape(feat1)[2]
            upsample_feat2 = tf.image.resize_bilinear(feat2,
                                                      size=[h, w],
                                                      name='up_sample_' + scope)

            add_f = upsample_feat2 + feat1
            reduce_dim_f = slim.conv2d(add_f,
                                       num_outputs=self.cfgs.FPN_CHANNEL,
                                       kernel_size=[1, 1], stride=1,
                                       trainable=is_training,
                                       scope='reduce_dim_' + scope)

            return reduce_dim_f

    def build_inception(self, inputs, is_training):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 384, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=self.cfgs.INITIALIZER,
                                       activation_fn=tf.nn.relu,
                                       scope='conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 192, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=self.cfgs.INITIALIZER,
                                       activation_fn=tf.nn.relu,
                                       scope='conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 224, [1, 7],
                                       trainable=is_training,
                                       weights_initializer=self.cfgs.INITIALIZER,
                                       activation_fn=tf.nn.relu,
                                       scope='conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 256, [7, 1],
                                       trainable=is_training,
                                       weights_initializer=self.cfgs.INITIALIZER,
                                       activation_fn=tf.nn.relu,
                                       scope='conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs, 192, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=self.cfgs.INITIALIZER,
                                       activation_fn=tf.nn.relu,
                                       scope='conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 192, [7, 1],
                                       trainable=is_training,
                                       weights_initializer=self.cfgs.INITIALIZER,
                                       activation_fn=tf.nn.relu,
                                       scope='conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, 224, [1, 7],
                                       trainable=is_training,
                                       weights_initializer=self.cfgs.INITIALIZER,
                                       activation_fn=tf.nn.relu,
                                       scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, 224, [7, 1],
                                       trainable=is_training,
                                       weights_initializer=self.cfgs.INITIALIZER,
                                       activation_fn=tf.nn.relu,
                                       scope='conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, 256, [1, 7],
                                       trainable=is_training,
                                       weights_initializer=self.cfgs.INITIALIZER,
                                       activation_fn=tf.nn.relu,
                                       scope='conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='avgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 128, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=self.cfgs.INITIALIZER,
                                       activation_fn=tf.nn.relu,
                                       scope='conv2d_0b_1x1')
            inception_out = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            return inception_out

    def build_inception_attention(self, inputs, is_training):
        """Builds Inception-B block for Inception v4 network."""
        # By default use stride=1 and SAME padding
        inception_out = self.build_inception(inputs, is_training)

        inception_attention_out = slim.conv2d(inception_out, 2, [3, 3],
                                              trainable=is_training,
                                              weights_initializer=self.cfgs.INITIALIZER,
                                              activation_fn=None,
                                              scope='inception_attention_out')
        return inception_attention_out

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name, is_training):
        with tf.name_scope(layer_name):
            # Global_Average_Pooling
            squeeze = tf.reduce_mean(input_x, [1, 2])

            excitation = slim.fully_connected(inputs=squeeze,
                                              num_outputs=out_dim // ratio,
                                              weights_initializer=self.cfgs.BBOX_INITIALIZER,
                                              activation_fn=tf.nn.relu,
                                              trainable=is_training,
                                              scope=layer_name + '_fully_connected1')

            excitation = slim.fully_connected(inputs=excitation,
                                              num_outputs=out_dim,
                                              weights_initializer=self.cfgs.BBOX_INITIALIZER,
                                              activation_fn=tf.nn.sigmoid,
                                              trainable=is_training,
                                              scope=layer_name + '_fully_connected2')

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

            # scale = input_x * excitation

            return excitation

    def sfnet(self, feature_dict, is_training):
        C3 = feature_dict['C3']

        h, w = tf.shape(C3)[1], tf.shape(C3)[2]
        resize_scale = 8 // self.cfgs.ANCHOR_STRIDE
        h_resize, w_size = h * resize_scale, w * resize_scale

        upsample_c3 = tf.image.resize_bilinear(C3,
                                               size=[h_resize, w_size],
                                               name='up_sample_c3')
        upsample_c3 = self.build_inception(upsample_c3, is_training)

        C4 = feature_dict['C4']
        out = self.fusion_two_layer(upsample_c3, C4, 'c3c4', is_training)
        return out

    def mdanet(self, feature, is_training):
        with tf.variable_scope('build_attention',
                               regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY)):
            ca = self.squeeze_excitation_layer(feature, self.cfgs.FPN_CHANNEL, 16, 'SE', is_training)

            pa_mask = self.build_inception_attention(feature, is_training)
            pa_mask_softmax = tf.nn.softmax(pa_mask)
            pa = pa_mask_softmax[:, :, :, 0]
            pa = tf.expand_dims(pa, axis=-1)

            out = tf.multiply(pa, feature)
            out *= ca

            return out, pa_mask

    def fpn(self, feature_dict, is_training):
        sfnet_feat = self.sfnet(feature_dict, is_training)
        mdanet_feat, pa_mask = self.mdanet(sfnet_feat, is_training)
        return mdanet_feat, pa_mask
