# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim


class ResNetGluonCVBackbone(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs

        self.DATA_FORMAT = "NHWC"
        self.DEBUG = False
        self.debug_dict = {}
        self.BottleNeck_NUM_DICT = {
            'resnet18_v1b': [2, 2, 2, 2],
            'resnet34_v1b': [3, 4, 6, 3],
            'resnet50_v1b': [3, 4, 6, 3],
            'resnet101_v1b': [3, 4, 23, 3],
            'resnet152_v1b': [3, 8, 36, 3],
            'resnet50_v1d': [3, 4, 6, 3],
            'resnet101_v1d': [3, 4, 23, 3],
            'resnet152_v1d': [3, 8, 36, 3]
        }

        self.BASE_CHANNELS_DICT = {
            'resnet18_v1b': [64, 128, 256, 512],
            'resnet34_v1b': [64, 128, 256, 512],
            'resnet50_v1b': [64, 128, 256, 512],
            'resnet101_v1b': [64, 128, 256, 512],
            'resnet152_v1b': [64, 128, 256, 512],
            'resnet50_v1d': [64, 128, 256, 512],
            'resnet101_v1d': [64, 128, 256, 512],
            'resnet152_v1d': [64, 128, 256, 512]
        }

    def resnet_arg_scope(self, freeze_norm, is_training=True, weight_decay=0.0001,
                         batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True):

        batch_norm_params = {
            'is_training': False, 'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
            'trainable': False,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'data_format': self.DATA_FORMAT
        }
        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer(),
                trainable=is_training,
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
                return arg_sc

    def stem_7x7(self, net, scope="C1"):

        with tf.variable_scope(scope):
            net = tf.pad(net, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])  # pad for data
            net = slim.conv2d(net, num_outputs=64, kernel_size=[7, 7], stride=2,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope="conv0")
            if self.DEBUG:
                self.debug_dict['conv_7x7_bn_relu'] = tf.transpose(net, [0, 3, 1, 2])  # NHWC --> NCHW
            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID", data_format=self.DATA_FORMAT)
            return net

    def stem_stack_3x3(self, net, input_channel=32, scope="C1"):
        with tf.variable_scope(scope):
            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=2,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope='conv0')
            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=1,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope='conv1')
            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.conv2d(net, num_outputs=input_channel*2, kernel_size=[3, 3], stride=1,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope='conv2')
            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID", data_format=self.DATA_FORMAT)
            return net

    def basicblock_v1b(self, input_x, base_channel, scope, stride=1, projection=False, avg_down=True, rm_shortcut=False):

        with tf.variable_scope(scope):
            if self.DEBUG:
                self.debug_dict[input_x.op.name] = tf.transpose(input_x, [0, 3, 1, 2])

            net = tf.pad(input_x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.conv2d(net, num_outputs=base_channel, kernel_size=[3, 3], stride=stride,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope='conv0')
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])

            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.conv2d(net, num_outputs=base_channel, kernel_size=[3, 3], stride=1,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              activation_fn=None, scope='conv1')
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
            # Note that : gamma in the last conv should be init with 0.
            # But we just reload params from mxnet, so don't specific batch norm initializer
            if projection:

                if avg_down:  # design for resnet_v1d
                    '''
                    In GluonCV, padding is "ceil mode". Here we use "SAME" to replace it, which may cause Erros.
                    And the erro will grow with depth of resnet. e.g. res101 erro > res50 erro
                    '''
                    shortcut = slim.avg_pool2d(input_x, kernel_size=[stride, stride], stride=stride, padding="SAME",
                                               data_format=self.DATA_FORMAT)
                    if self.DEBUG:
                        self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])

                    shortcut = slim.conv2d(shortcut, num_outputs=base_channel, kernel_size=[1, 1],
                                           stride=1, padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                                           activation_fn=None,
                                           scope='shortcut')
                    if self.DEBUG:
                        self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])
                        # shortcut should have batch norm.
                else:
                    if rm_shortcut:
                        shortcut = tf.identity(input_x, name='shortcut/Identity')
                    else:
                        shortcut = slim.conv2d(input_x, num_outputs=base_channel, kernel_size=[1, 1],
                                               stride=stride, padding="VALID", biases_initializer=None,
                                               activation_fn=None,
                                               data_format=self.DATA_FORMAT,
                                               scope='shortcut')
                    if self.DEBUG:
                        self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])
            else:
                shortcut = tf.identity(input_x, name='shortcut/Identity')
                if self.DEBUG:
                    self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])

            net = net + shortcut
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
            net = tf.nn.relu(net)
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
            return net

    def bottleneck_v1b(self, input_x, base_channel, scope, stride=1, projection=False, avg_down=True):
        '''
        for bottleneck_v1b: reduce spatial dim in conv_3x3 with stride 2.
        '''
        with tf.variable_scope(scope):
            if self.DEBUG:
                self.debug_dict[input_x.op.name] = tf.transpose(input_x, [0, 3, 1, 2])
            net = slim.conv2d(input_x, num_outputs=base_channel, kernel_size=[1, 1], stride=1,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope='conv0')
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])

            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])

            net = slim.conv2d(net, num_outputs=base_channel, kernel_size=[3, 3], stride=stride,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope='conv1')
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
            net = slim.conv2d(net, num_outputs=base_channel * 4, kernel_size=[1, 1], stride=1,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              activation_fn=None, scope='conv2')
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
            # Note that : gamma in the last conv should be init with 0.
            # But we just reload params from mxnet, so don't specific batch norm initializer
            if projection:

                if avg_down:  # design for resnet_v1d
                    '''
                    In GluonCV, padding is "ceil mode". Here we use "SAME" to replace it, which may cause Erros.
                    And the erro will grow with depth of resnet. e.g. res101 erro > res50 erro
                    '''
                    shortcut = slim.avg_pool2d(input_x, kernel_size=[stride, stride], stride=stride, padding="SAME",
                                               data_format=self.DATA_FORMAT)
                    if self.DEBUG:
                        self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])

                    shortcut = slim.conv2d(shortcut, num_outputs=base_channel*4, kernel_size=[1, 1],
                                           stride=1, padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                                           activation_fn=None,
                                           scope='shortcut')
                    if self.DEBUG:
                        self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])
                    # shortcut should have batch norm.
                else:
                    shortcut = slim.conv2d(input_x, num_outputs=base_channel * 4, kernel_size=[1, 1],
                                           stride=stride, padding="VALID", biases_initializer=None, activation_fn=None,
                                           data_format=self.DATA_FORMAT,
                                           scope='shortcut')
                    if self.DEBUG:
                        self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])
            else:
                shortcut = tf.identity(input_x, name='shortcut/Identity')
                if self.DEBUG:
                    self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])

            net = net + shortcut
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
            net = tf.nn.relu(net)
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
            return net

    def make_block_(self, net, base_channel, bottleneck_nums, scope, avg_down=True, spatial_downsample=False):
        with tf.variable_scope(scope):
            first_stride = 2 if spatial_downsample else 1
            if scope == 'C2':
                rm_shortcut = True
            else:
                rm_shortcut = False

            net = self.basicblock_v1b(input_x=net, base_channel=base_channel, scope='basicblock_0',
                                      stride=first_stride, avg_down=avg_down, projection=True, rm_shortcut=rm_shortcut)
            for i in range(1, bottleneck_nums):
                net = self.basicblock_v1b(input_x=net, base_channel=base_channel, scope="basicblock_%d" % i,
                                          stride=1, avg_down=avg_down, projection=False)
            return net

    def make_block(self, net, base_channel, bottleneck_nums, scope, avg_down=True, spatial_downsample=False):
        with tf.variable_scope(scope):
            first_stride = 2 if spatial_downsample else 1

            net = self.bottleneck_v1b(input_x=net, base_channel=base_channel,scope='bottleneck_0',
                                      stride=first_stride, avg_down=avg_down, projection=True)
            for i in range(1, bottleneck_nums):
                net = self.bottleneck_v1b(input_x=net, base_channel=base_channel, scope="bottleneck_%d" % i,
                                          stride=1, avg_down=avg_down, projection=False)
            return net

    def get_resnet_v1_b_base(self, input_x, freeze_norm, scope="resnet50_v1b", bottleneck_nums=[3, 4, 6, 3], base_channels=[64, 128, 256, 512],
                        freeze=[True, False, False, False, False], is_training=True):

        assert len(bottleneck_nums) == len(base_channels), "bottleneck num should same as base_channels size"
        assert len(freeze) == len(bottleneck_nums) +1, "should satisfy:: len(freeze) == len(bottleneck_nums) + 1"
        feature_dict = {}
        with tf.variable_scope(scope):
            with slim.arg_scope(self.resnet_arg_scope(is_training=(not freeze[0]) and is_training,
                                                      freeze_norm=freeze_norm)):
                net = self.stem_7x7(net=input_x, scope="C1")
                feature_dict["C1"] = net
            for i in range(2, len(bottleneck_nums)+2):
                spatial_downsample = False if i == 2 else True
                with slim.arg_scope(self.resnet_arg_scope(is_training=(not freeze[i-1]) and is_training,
                                                          freeze_norm=freeze_norm)):
                    if '34' in scope or '18' in scope:
                        net = self.make_block_(net=net, base_channel=base_channels[i - 2],
                                               bottleneck_nums=bottleneck_nums[i - 2],
                                               scope="C%d" % i,
                                               avg_down=False, spatial_downsample=spatial_downsample)
                    else:
                        net = self.make_block(net=net, base_channel=base_channels[i-2],
                                              bottleneck_nums=bottleneck_nums[i-2],
                                              scope="C%d" % i,
                                              avg_down=False, spatial_downsample=spatial_downsample)
                    feature_dict["C%d" % i] = net

        return net, feature_dict

    def get_resnet_v1_d_base(self, input_x, freeze_norm, scope="resnet50_v1d", bottleneck_nums=[3, 4, 6, 3], base_channels=[64, 128, 256, 512],
                        freeze=[True, False, False, False, False], is_training=True):

        assert len(bottleneck_nums) == len(base_channels), "bottleneck num should same as base_channels size"
        assert len(freeze) == len(bottleneck_nums) + 1, "should satisfy:: len(freeze) == len(bottleneck_nums) + 1"
        feature_dict = {}
        with tf.variable_scope(scope):
            with slim.arg_scope(self.resnet_arg_scope(is_training=((not freeze[0]) and is_training),
                                                      freeze_norm=freeze_norm)):
                net = self.stem_stack_3x3(net=input_x, input_channel=32, scope="C1")
                feature_dict["C1"] = net
                # print (net)
            for i in range(2, len(bottleneck_nums)+(2 if self.cfgs.FPN_MODE != 'scrdet' else 1)):
                spatial_downsample = False if i == 2 else True  # do not downsample in C2
                with slim.arg_scope(self.resnet_arg_scope(is_training=((not freeze[i-1]) and is_training),
                                                          freeze_norm=freeze_norm)):

                    net = self.make_block(net=net, base_channel=base_channels[i-2],
                                          bottleneck_nums=bottleneck_nums[i-2],
                                          scope="C%d" % i,
                                          avg_down=True, spatial_downsample=spatial_downsample)
                    feature_dict["C%d" % i] = net

        return net, feature_dict

    # -----------------------------------
    def resnet_base(self, img_batch, scope_name, is_training=True):
        if scope_name.endswith('b'):
            get_resnet_fn = self.get_resnet_v1_b_base
        elif scope_name.endswith('d'):
            get_resnet_fn = self.get_resnet_v1_d_base
        else:
            raise ValueError("scope Name erro....")

        _, feature_dict = get_resnet_fn(input_x=img_batch, scope=scope_name,
                                        bottleneck_nums=self.BottleNeck_NUM_DICT[scope_name],
                                        base_channels=self.BASE_CHANNELS_DICT[scope_name],
                                        is_training=is_training, freeze_norm=True,
                                        freeze=self.cfgs.FREEZE_BLOCKS)

        return feature_dict


