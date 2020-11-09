# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import tensorflow.contrib.slim as slim
import tensorflow as tf


class NeckFPN(object):
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
                                       num_outputs=256,
                                       kernel_size=[1, 1], stride=1,
                                       trainable=is_training,
                                       scope='reduce_dim_'+level_name)

            add_f = 0.5*upsample_p + 0.5*reduce_dim_c

            return add_f

    def fpn(self, feature_dict, is_training):

        pyramid_dict = {}
        with tf.variable_scope('build_pyramid'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.cfgs.WEIGHT_DECAY),
                                activation_fn=None, normalizer_fn=None):

                P5 = slim.conv2d(feature_dict['C5'],
                                 num_outputs=256,
                                 kernel_size=[1, 1],
                                 stride=1,
                                 trainable=is_training,
                                 scope='build_P5')

                if self.cfgs.ADD_GLOBAL_CTX:
                    print(10 * "ADD GLOBAL CTX.....")
                    global_ctx = tf.reduce_mean(feature_dict['C5'], axis=[1, 2], keep_dims=True)
                    global_ctx = slim.conv2d(global_ctx, kernel_size=[1, 1], num_outputs=256, stride=1,
                                             activation_fn=None, trainable=is_training, scope='global_ctx')
                    pyramid_dict['P5'] = P5 + global_ctx
                else:
                    pyramid_dict['P5'] = P5

                for level in range(4, int(self.cfgs.LEVEL[0][-1]) - 1, -1):  # build [P4, P3, P2]

                    pyramid_dict['P%d' % level] = self.fusion_two_layer(C_i=feature_dict["C%d" % level],
                                                                        P_j=pyramid_dict["P%d" % (level + 1)],
                                                                        scope='build_P%d' % level,
                                                                        is_training=is_training)
                for level in range(5, int(self.cfgs.LEVEL[0][-1]) - 1, -1):  # use 3x3 conv fuse P5, P4, P3, P2
                    pyramid_dict['P%d' % level] = slim.conv2d(pyramid_dict['P%d' % level],
                                                              num_outputs=256, kernel_size=[3, 3], padding="SAME",
                                                              stride=1, trainable=is_training,
                                                              scope="fuse_P%d" % level)
                if "P6" in self.cfgs.LEVEL:
                    # if use supervised_mask, we get p6 after enlarge RF
                    pyramid_dict['P6'] = slim.avg_pool2d(pyramid_dict["P5"], kernel_size=[2, 2],
                                                         stride=2, scope='build_P6')
        # for level in range(5, 1, -1):
        #     add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/P%d_fpn_heat' % (level, level))

        return pyramid_dict