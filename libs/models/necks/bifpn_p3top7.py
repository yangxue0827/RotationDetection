# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class NeckBiFPNRetinaNet(object):
    """
    copy from https://github.com/MingtaoGuo/RetinaNet_BiFPN_TensorFlow/blob/master/ops.py
    """
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.W_bifpn = 64
        self.D_bifpn = 3
        self.EPSILON = 1e-8

    def batchnorm(self, scope_bn, x, train_phase, epsilon=0.001, decay=0.99):
        """
        Performs a batch normalization layer
        Args:
            x: input tensor
            scope: scope name
            is_training: python boolean value
            epsilon: the variance epsilon - a small float number to avoid dividing by 0
            decay: the moving average decay
        Returns:
            The ops of a batch normalization layer
        """
        train_phase = True
        with tf.variable_scope(scope_bn):
            shape = x.get_shape().as_list()
            # gamma: a trainable scale factor
            gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
            # beta: a trainable shift value
            beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
            moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0),
                                         trainable=False)
            if train_phase:
                # tf.nn.moments == Calculate the mean and the variance of the tensor x
                avg, var = tf.nn.moments(x, np.arange(len(shape) - 1), keep_dims=True)
                avg = tf.reshape(avg, [avg.shape.as_list()[-1]])
                var = tf.reshape(var, [var.shape.as_list()[-1]])
                # update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
                update_moving_avg = tf.assign(moving_avg, moving_avg * decay + avg * (1 - decay))
                # update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
                update_moving_var = tf.assign(moving_var, moving_var * decay + var * (1 - decay))
                control_inputs = [update_moving_avg, update_moving_var]
            else:
                avg = moving_avg
                var = moving_var
                control_inputs = []
            with tf.control_dependencies(control_inputs):
                output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

        return output

    def conv(self, name, inputs, nums_out, k_size, stride, padding, is_final=False):
        nums_in = inputs.shape[-1]
        with tf.variable_scope(name):
            W = tf.get_variable("W", [k_size, k_size, nums_in, nums_out],
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
            if is_final:
                b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([-np.log((1 - 0.01) / 0.01)]))
            else:
                b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
            inputs = tf.nn.conv2d(inputs, W, [1, stride, stride, 1], padding)
            inputs = tf.nn.bias_add(inputs, b)
        return inputs

    def swish(self, inputs):
        return tf.nn.swish(inputs)

    def sigmoid(self, inputs):
        return tf.nn.sigmoid(inputs)

    def resize(self, inputs, factor=2.):
        H, W = int(inputs.shape[1]), int(inputs.shape[2])
        return tf.image.resize_nearest_neighbor(inputs, [int(H * factor), int(W * factor)])

    def fusion_two_layer(self, C_i, P_j, scope):
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

            add_feat = upsample_p + C_i

            return add_feat

    def bifpn_layer(self, name, p3, p4, p5, p6, p7, train_phase):
        with tf.variable_scope(name):
            with tf.variable_scope("intermediate_p6"):
                w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                # temp = w1 * p6 + w2 * self.resize(p7)
                temp = self.fusion_two_layer(w1 * p6, w2 * p7, 'p6_p7')
                P_6_td = self.swish(self.batchnorm("bn1",
                                         self.conv("conv6_td", temp / (w1 + w2 + self.EPSILON), self.W_bifpn, k_size=3, stride=1,
                                              padding="SAME"), train_phase))
            with tf.variable_scope("intermediate_p5"):
                w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                # temp = w1 * p5 + w2 * self.resize(P_6_td)
                temp = self.fusion_two_layer(w1 * p5, w2 * P_6_td, 'p5_P_6_td')
                P_5_td = self.swish(self.batchnorm("bn1",
                                         self.conv("conv5_td", temp / (w1 + w2 + self.EPSILON), self.W_bifpn, k_size=3, stride=1,
                                              padding="SAME"), train_phase))
            with tf.variable_scope("intermediate_p4"):
                w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                # temp = w1 * p4 + w2 * self.resize(P_5_td)
                temp = self.fusion_two_layer(w1 * p4, w2 * P_5_td, 'p4_P_5_td')
                P_4_td = self.swish(self.batchnorm("bn1",
                                         self.conv("conv4_td", temp / (w1 + w2 + self.EPSILON), self.W_bifpn, k_size=3, stride=1,
                                              padding="SAME"), train_phase))
            with tf.variable_scope("output_p3"):
                w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                # temp = w1 * p3 + w2 * self.resize(P_4_td)
                temp = self.fusion_two_layer(w1 * p3, w2 * P_4_td, 'p3_P_4_td')
                P_3_out = self.swish(self.batchnorm("bn1",
                                          self.conv("conv3_out", temp / (w1 + w2 + self.EPSILON), self.W_bifpn, k_size=3, stride=1,
                                               padding="SAME"), train_phase))
            with tf.variable_scope("output_p4"):
                w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                w3 = tf.get_variable("W3", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                # temp = w1 * p4 + w2 * P_4_td + w3 * self.resize(P_3_out, factor=0.5)
                temp = self.fusion_two_layer(w1 * p4 + w2 * P_4_td, w3 * P_3_out, 'p4_P_4_td_P_3_out')
                P_4_out = self.swish(self.batchnorm("bn1", self.conv("conv4_out", temp / (w1 + w2 + w3 + self.EPSILON), self.W_bifpn, k_size=3,
                                                      stride=1, padding="SAME"), train_phase))
            with tf.variable_scope("output_p5"):
                w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                w3 = tf.get_variable("W3", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                # temp = w1 * p5 + w2 * P_5_td + w3 * self.resize(P_4_out, factor=0.5)
                temp = self.fusion_two_layer(w1 * p5 + w2 * P_5_td, w3 * P_4_out, 'p5_P_5_td_P_4_out')
                P_5_out = self.swish(self.batchnorm("bn1", self.conv("conv5_out", temp / (w1 + w2 + w3 + self.EPSILON), self.W_bifpn, k_size=3,
                                                      stride=1, padding="SAME"), train_phase))
            with tf.variable_scope("output_p6"):
                w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                w3 = tf.get_variable("W3", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                # temp = w1 * p6 + w2 * P_6_td + w3 * self.resize(P_5_out, factor=0.5)
                temp = self.fusion_two_layer(w1 * p6 + w2 * P_6_td, w3 * P_5_out, 'p6_P_6_td_P_5_out')
                P_6_out = self.swish(self.batchnorm("bn1", self.conv("conv6_out", temp / (w1 + w2 + w3 + self.EPSILON), self.W_bifpn, k_size=3,
                                                      stride=1, padding="SAME"), train_phase))
            with tf.variable_scope("output_p7"):
                w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
                # temp = w1 * p7 + w2 * self.resize(P_6_out, factor=0.5)
                temp = self.fusion_two_layer(w1 * p7, w2 * P_6_out, 'p7_P_6_out')
                P_7_out = self.swish(self.batchnorm("bn1",
                                          self.conv("conv7_out", temp / (w1 + w2 + self.EPSILON), self.W_bifpn, k_size=3, stride=1,
                                               padding="SAME"), train_phase))
        return P_3_out, P_4_out, P_5_out, P_6_out, P_7_out

    def fpn_retinanet(self, feature_dict, is_training):
        C3 = feature_dict['C3']
        C4 = feature_dict['C4']
        C5 = feature_dict['C5']
        P3 = self.swish(self.batchnorm("bn1", self.conv("conv3", C3, self.W_bifpn, 3, 1, "SAME"), is_training))
        P4 = self.swish(self.batchnorm("bn2", self.conv("conv4", C4, self.W_bifpn, 3, 1, "SAME"), is_training))
        P5 = self.swish(self.batchnorm("bn3", self.conv("conv5", C5, self.W_bifpn, 3, 1, "SAME"), is_training))
        P6 = self.swish(self.batchnorm("bn4", self.conv("conv6", C5, self.W_bifpn, 3, 2, "SAME"), is_training))
        P7 = self.swish(self.batchnorm("bn5", self.conv("conv7", P6, self.W_bifpn, 3, 2, "SAME"), is_training))
        for i in range(self.D_bifpn):
            P3, P4, P5, P6, P7 = self.bifpn_layer("bifpn" + str(i), P3, P4, P5, P6, P7, is_training)

        pyramid_dict = {'P3': P3, 'P4': P4, 'P5': P5, 'P6': P6, 'P7': P7}
        return pyramid_dict

