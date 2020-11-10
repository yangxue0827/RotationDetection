# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf

slim = tf.contrib.slim


class DarkNetBackbone(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs

    def conv2d(self, inputs, filters, kernel_size, strides=1):
        def _fixed_padding(inputs, kernel_size):
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg

            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                            [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
            return padded_inputs

        if strides > 1:
            inputs = _fixed_padding(inputs, kernel_size)
        inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                             padding=('SAME' if strides == 1 else 'VALID'))
        return inputs

    def darknet53_body(self, inputs, is_training):

        batch_norm_params = {
            'decay': 0.999,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': True,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm]):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(5e-4)):
                with tf.variable_scope('darknet/darknet53_body'):
                    def res_block(inputs, filters):
                        shortcut = inputs
                        net = self.conv2d(inputs, filters * 1, 1)
                        net = self.conv2d(net, filters * 2, 3)

                        net = net + shortcut

                        return net

                    # first two conv2d layers
                    net = self.conv2d(inputs, 32, 3, strides=1)
                    net = self.conv2d(net, 64, 3, strides=2)
                    # res_block * 1
                    net = res_block(net, 32)

                    net = self.conv2d(net, 128, 3, strides=2)
                    # res_block * 2
                    for i in range(2):
                        net = res_block(net, 64)
                    route_0 = net

                    net = self.conv2d(net, 256, 3, strides=2)
                    # res_block * 8
                    for i in range(8):
                        net = res_block(net, 128)

                    route_1 = net
                    net = self.conv2d(net, 512, 3, strides=2)
                    # res_block * 8
                    for i in range(8):
                        net = res_block(net, 256)

                    route_2 = net
                    net = self.conv2d(net, 1024, 3, strides=2)

                    # res_block * 4
                    for i in range(4):
                        net = res_block(net, 512)
                    route_3 = net

                    feature_dict = {'C2': route_0,
                                    'C3': route_1,
                                    'C4': route_2,
                                    'C5': route_3,
                                    }

                    return feature_dict


if __name__ == '__main__':
    from libs.configs import cfgs
    inputs = tf.placeholder(tf.float32, [1, 416, 416, 3])
    darknet = DarkNetBackbone(cfgs)
    print(darknet.darknet53_body(inputs, False))