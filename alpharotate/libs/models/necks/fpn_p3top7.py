# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import tensorflow.contrib.slim as slim
import tensorflow as tf


class NeckFPNRetinaNet(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def fusion_two_layer(self, C_i, P_j, scope, is_training):
        '''
        i = j+1
        :param C_i: shape is [1, h, w, c]
        :param P_j: shape is [1, h/2, w/2, 256]
        :return:
        P_i
        '''
        with tf.variable_scope(scope):
            level_name = scope.split('_')[1]

            h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
            upsample_p = tf.image.resize_bilinear(P_j,
                                                  size=[h, w],
                                                  name='up_sample_'+level_name)

            reduce_dim_c = slim.conv2d(C_i,
                                       num_outputs=self.cfgs.FPN_CHANNEL,
                                       kernel_size=[1, 1], stride=1,
                                       trainable=is_training,
                                       scope='reduce_dim_'+level_name)

            add_f = 0.5*upsample_p + 0.5*reduce_dim_c

            # P_i = slim.conv2d(add_f,
            #                   num_outputs=256, kernel_size=[3, 3], stride=1,
            #                   padding='SAME',
            #                   scope='fusion_'+level_name)
            return add_f

    def fpn_retinanet(self, feature_dict, is_training):

        pyramid_dict = {}
        with tf.variable_scope('build_pyramid'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY),
                                activation_fn=None, normalizer_fn=None):

                P5 = slim.conv2d(feature_dict['C5'],
                                 num_outputs=self.cfgs.FPN_CHANNEL,
                                 kernel_size=[1, 1],
                                 trainable=is_training,
                                 stride=1, scope='build_P5')

                pyramid_dict['P5'] = P5

                for level in range(4, int(self.cfgs.LEVEL[0][-1]) - 1, -1):  # build [P4, P3]

                    pyramid_dict['P%d' % level] = self.fusion_two_layer(C_i=feature_dict["C%d" % level],
                                                                        P_j=pyramid_dict["P%d" % (level + 1)],
                                                                        scope='build_P%d' % level,
                                                                        is_training=is_training)
                for level in range(5, int(self.cfgs.LEVEL[0][-1]) - 1, -1):
                    pyramid_dict['P%d' % level] = slim.conv2d(pyramid_dict['P%d' % level],
                                                              num_outputs=self.cfgs.FPN_CHANNEL, kernel_size=[3, 3],
                                                              padding="SAME",
                                                              stride=1, scope="fuse_P%d" % level)

                p6 = slim.conv2d(pyramid_dict['P5'] if self.cfgs.USE_P5 else feature_dict['C5'],
                                 num_outputs=self.cfgs.FPN_CHANNEL, kernel_size=[3, 3], padding="SAME",
                                 stride=2, trainable=is_training, scope='p6_conv')
                pyramid_dict['P6'] = p6

                if int(self.cfgs.LEVEL[-1][-1]) == 7:

                    p7 = tf.nn.relu(p6, name='p6_relu')

                    p7 = slim.conv2d(p7,
                                     num_outputs=self.cfgs.FPN_CHANNEL, kernel_size=[3, 3], padding="SAME",
                                     stride=2, trainable=is_training, scope='p7_conv')

                    pyramid_dict['P7'] = p7

        # for level in range(7, 1, -1):
        #     add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/P%d_heat' % (level, level))

        return pyramid_dict
