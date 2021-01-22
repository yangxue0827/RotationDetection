# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet.gluon import nn
from mxnet import ndarray as nd
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
from test_resnet import mxnet_process_img
# 卷积层
# 输入输出的数据格式是： batch * channel * height * width
# 权重格式：output_channels * in_channels * height * width
np.random.seed(30)

# w = nd.array(np.random.rand(2, 3, 3, 3))
w = nd.load('/home/yjr/MxNet_Codes/gluon-cv/scripts/gloun2TF/mxnet_weights/resnet50_v1b-0ecdba34.params')['conv1.weight']  # [64, 3, 7, 7]
# w = nd.arange(9*2).reshape((2, 1, 3, 3))
data = nd.array(np.random.rand(1, 3, 224, 224))
# data, _ = mxnet_process_img('../demo_img/person.jpg')
# data = nd.arange(6*6).reshape((1, 1, 6, 6))

# 卷积运算
out = nd.Convolution(data, w, no_bias=True,
                     kernel=(7, 7),
                     stride=(2, 2),
                     num_filter=64,
                     pad=(3, 3))



def tf_conv(data, w):

    data = tf.constant(data.asnumpy())
    data = tf.pad(data, paddings=[[0, 0], [0, 0], [3, 3], [3, 3]])
    tf_out = slim.conv2d(data, num_outputs=64, kernel_size=[7, 7], padding='VALID', stride=2,
                         biases_initializer=None, data_format='NCHW', normalizer_fn=None, activation_fn=None)
    tf_w = tf.constant(np.transpose(w.asnumpy(), [2, 3, 1, 0]))
    # tf_w =
    model_vars = slim.get_model_variables()
    assign_op = tf.assign(model_vars[0], tf_w)

    with tf.Session() as sess:
        sess.run(assign_op)
        print(sess.run(tf_out))


if __name__ == '__main__':
    tf_conv(data, w=w)
    print "mxnet_out: ", out
    print 20 * "+"