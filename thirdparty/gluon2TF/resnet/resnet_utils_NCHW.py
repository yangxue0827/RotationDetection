# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math

DATA_FORMAT = "NCHW"  # to match data format for mxnet
debug_dict = {}
def resnet_arg_scope(freeze_norm, is_training=True, weight_decay=0.0001,
                     batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True):

    batch_norm_params = {
        'is_training': (not freeze_norm) and is_training, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': (not freeze_norm) and is_training,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'data_format': DATA_FORMAT
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


def stem_7x7(net, scope="C1"):

    with tf.variable_scope(scope):
        net = tf.pad(net, paddings=[[0, 0], [0, 0], [3, 3], [3, 3]])  # pad for data
        net = slim.conv2d(net, num_outputs=64, kernel_size=[7, 7], stride=2,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope="conv0", normalizer_fn=None, activation_fn=None)
        debug_dict['conv_7x7'] = net
        with tf.variable_scope('conv0') as scope:
            net = slim.batch_norm(net)
            debug_dict['conv_7x7_bn'] = net
            net = tf.nn.relu(net)
            debug_dict['conv_7x7_bn_relu'] = net

        net = tf.pad(net, paddings=[[0, 0], [0, 0], [1, 1], [1, 1]])
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID", data_format=DATA_FORMAT)
        return net

def stem_stack_3x3(net, input_channel=32, scope="C1"):
    with tf.variable_scope(scope):
        net = tf.pad(net, paddings=[[0, 0], [0, 0], [1, 1], [1, 1]])
        net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=2,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv0')
        net = tf.pad(net, paddings=[[0, 0], [0, 0], [1, 1], [1, 1]])
        net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv1')
        net = tf.pad(net, paddings=[[0, 0], [0, 0], [1, 1], [1, 1]])
        net = slim.conv2d(net, num_outputs=input_channel*2, kernel_size=[3, 3], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv2')
        net = tf.pad(net, paddings=[[0, 0], [0, 0], [1, 1], [1, 1]])
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID", data_format=DATA_FORMAT)
        return net


def bottleneck_v1b(input_x, base_channel, scope, stride=1, projection=False, avg_down=True):
    '''
    for bottleneck_v1b: reduce spatial dim in conv_3x3 with stride 2.
    '''
    with tf.variable_scope(scope):
        debug_dict[input_x.op.name] = input_x
        net = slim.conv2d(input_x, num_outputs=base_channel, kernel_size=[1, 1], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv0')
        debug_dict[net.op.name] = net
        net = tf.pad(net, paddings=[[0, 0], [0, 0], [1, 1], [1, 1]])
        debug_dict[net.op.name] = net
        net = slim.conv2d(net, num_outputs=base_channel, kernel_size=[3, 3], stride=stride,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv1')
        debug_dict[net.op.name] = net
        net = slim.conv2d(net, num_outputs=base_channel * 4, kernel_size=[1, 1], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          activation_fn=None, scope='conv2')
        debug_dict[net.op.name] = net
        # Note that : gamma in the last conv should be init with 0.
        # But we just reload params from mxnet, so don't specific batch norm initializer
        if projection:

            if avg_down:  # design for resnet_v1d
                # pad = 1  # int(math.floor((stride - 1)/2.0))
                # input_x = tf.pad(input_x, paddings=[[0, 0], [0, 0], [pad, pad], [pad, pad]])
                shortcut = slim.avg_pool2d(input_x, kernel_size=[stride, stride], stride=stride, padding="SAME",
                                           data_format=DATA_FORMAT)
                debug_dict[shortcut.op.name] = shortcut
                shortcut = slim.conv2d(shortcut, num_outputs=base_channel*4, kernel_size=[1, 1],
                                       stride=1, padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                                       activation_fn=None,
                                       scope='shortcut')
                debug_dict[shortcut.op.name] = shortcut
                # shortcut should have batch norm.
            else:
                shortcut = slim.conv2d(input_x, num_outputs=base_channel * 4, kernel_size=[1, 1],
                                       stride=stride, padding="VALID", biases_initializer=None, activation_fn=None,
                                       data_format=DATA_FORMAT,
                                       scope='shortcut')
                debug_dict[shortcut.op.name] = shortcut
        else:
            shortcut = tf.identity(input_x, name='shortcut/Relu')
            debug_dict[shortcut.op.name] = shortcut

        net = net + shortcut
        debug_dict[net.op.name] = net
        net = tf.nn.relu(net)
        debug_dict[net.op.name] = net
        return net


def make_block(net, base_channel, bottleneck_nums, scope, avg_down=True, spatial_downsample=False):
    with tf.variable_scope(scope):
        first_stride = 2 if spatial_downsample else 1

        net = bottleneck_v1b(input_x=net, base_channel=base_channel,scope='bottleneck_0',
                             stride=first_stride, avg_down=avg_down, projection=True)
        for i in range(1, bottleneck_nums):
            net = bottleneck_v1b(input_x=net, base_channel=base_channel, scope="bottleneck_%d" % i,
                                 stride=1, avg_down=avg_down, projection=False)
        return net


def get_resnet_v1_b_base(input_x, freeze_norm, scope="resnet50_v1b", bottleneck_nums=[3, 4, 6, 3], base_channels=[64, 128, 256, 512],
                    freeze=[True, False, False, False, False], is_training=True):

    assert len(bottleneck_nums) == len(base_channels), "bottleneck num should same as base_channels size"
    assert len(freeze) == len(bottleneck_nums) +1, "should satisfy:: len(freeze) == len(bottleneck_nums) + 1"
    feature_dict = {}
    with tf.variable_scope(scope):
        with slim.arg_scope(resnet_arg_scope(is_training=(not freeze[0]) and is_training,
                                             freeze_norm=freeze_norm)):
            net = stem_7x7(net=input_x, scope="C1")
            feature_dict["C1"] = net
        for i in range(2, len(bottleneck_nums)+2):
            spatial_downsample = False if i == 2 else True
            with slim.arg_scope(resnet_arg_scope(is_training=(not freeze[i-2]) and is_training,
                                                 freeze_norm=freeze_norm)):
                net = make_block(net=net, base_channel=base_channels[i-2],
                                 bottleneck_nums=bottleneck_nums[i-2],
                                 scope="C%d" % i,
                                 avg_down=False, spatial_downsample=spatial_downsample)
                feature_dict["C%d" % i] = net

    return net, feature_dict


def get_resnet_v1_b(input_x,
                    scope="resnet50_v1b", bottleneck_nums=[3, 4, 6, 3],
                    base_channels=[64, 128, 256, 512],
                    freeze=[True, False, False, False, False],
                    is_training=True, freeze_norm=False,
                    num_cls=1000, dropout=False):

    net, fet_dict = get_resnet_v1_b_base(input_x=input_x, scope=scope, bottleneck_nums=bottleneck_nums, base_channels=base_channels,
                                         freeze=freeze, is_training=is_training, freeze_norm=freeze_norm)
    with tf.variable_scope(scope):
        # net shape : [B, C, H, W]
        if DATA_FORMAT.strip() == "NCHW":
            net = tf.reduce_mean(net, axis=[2, 3], name="global_avg_pooling",
                                 keep_dims=True)  # [B, C, 1, 1]
        elif DATA_FORMAT.strip() == "NHWC":
            net = tf.reduce_mean(net, axis=[1, 2], name="global_avg_pooling",
                                 keep_dims=True)  # [B, 1, 1, C]
        else:
            raise ValueError("Data Format Erro...")

        net = slim.flatten(net, scope='flatten')
        if dropout:
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
        logits = slim.fully_connected(net, num_outputs=num_cls, activation_fn=None, scope='logits')
        return logits


def get_resnet_v1_d_base(input_x, freeze_norm, scope="resnet50_v1d", bottleneck_nums=[3, 4, 6, 3], base_channels=[64, 128, 256, 512],
                    freeze=[True, False, False, False, False], is_training=True):

    assert len(bottleneck_nums) == len(base_channels), "bottleneck num should same as base_channels size"
    assert len(freeze) == len(bottleneck_nums) +1, "should satisfy:: len(freeze) == len(bottleneck_nums) + 1"
    feature_dict = {}
    with tf.variable_scope(scope):
        with slim.arg_scope(resnet_arg_scope(is_training=(not freeze[0]) and is_training,
                                             freeze_norm=freeze_norm)):
            net = stem_stack_3x3(net=input_x, input_channel=32, scope="C1")
            feature_dict["C1"] = net
        for i in range(2, len(bottleneck_nums)+2):
            spatial_downsample = False if i == 2 else True
            with slim.arg_scope(resnet_arg_scope(is_training=(not freeze[i-2]) and is_training,
                                                 freeze_norm=freeze_norm)):
                net = make_block(net=net, base_channel=base_channels[i-2],
                                 bottleneck_nums=bottleneck_nums[i-2],
                                 scope="C%d" % i,
                                 avg_down=True, spatial_downsample=spatial_downsample)
                feature_dict["C%d" % i] = net

    return net, feature_dict

def get_resnet_v1_d(input_x,
                    scope="resnet50_v1d", bottleneck_nums=[3, 4, 6, 3],
                    base_channels=[64, 128, 256, 512],
                    freeze=[True, False, False, False, False],
                    is_training=True, freeze_norm=False,
                    num_cls=1000, dropout=False):

    net, fet_dict = get_resnet_v1_d_base(input_x=input_x, scope=scope, bottleneck_nums=bottleneck_nums, base_channels=base_channels,
                                         freeze=freeze, is_training=is_training, freeze_norm=freeze_norm)
    with tf.variable_scope(scope):
        # net shape : [B, C, H, W]
        if DATA_FORMAT.strip() == "NCHW":
            net = tf.reduce_mean(net, axis=[2, 3], name="global_avg_pooling",
                                 keep_dims=True)  # [B, C, 1, 1]
        elif DATA_FORMAT.strip() == "NHWC":
            net = tf.reduce_mean(net, axis=[1, 2], name="global_avg_pooling",
                                 keep_dims=True)  # [B, 1, 1, C]
        else:
            raise ValueError("Data Format Erro...")

        net = slim.flatten(net, scope='flatten')
        if dropout:
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
        logits = slim.fully_connected(net, num_outputs=num_cls, activation_fn=None, scope='logits')
        return logits






