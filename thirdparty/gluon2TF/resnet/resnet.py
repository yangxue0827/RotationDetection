# -*- coding: utf-8 -*-


from __future__ import absolute_import, print_function, division
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from resnet_utils import get_resnet_v1_d, get_resnet_v1_b, get_resnet_v1_s
from parse_mxnet_weights import read_mxnet_weights, check_mxnet_names, check_tf_vars
import weights_map
import os

BottleNeck_NUM_DICT = {
    'resnet18_v1b': [2, 2, 2, 2],
    'resnet34_v1b': [3, 4, 6, 3],
    'resnet50_v1b': [3, 4, 6, 3],
    'resnet101_v1b': [3, 4, 23, 3],
    'resnet50_v1d': [3, 4, 6, 3],
    'resnet101_v1d': [3, 4, 23, 3],
    'resnet152_v1d': [3, 8, 36, 3],
    'resnet50_v1s': [3, 4, 6, 3],
}

BASE_CHANNELS_DICT = {
    'resnet18_v1b': [64, 128, 256, 512],
    'resnet34_v1b': [64, 128, 256, 512],
    'resnet50_v1b': [64, 128, 256, 512],
    'resnet101_v1b': [64, 128, 256, 512],
    'resnet50_v1d': [64, 128, 256, 512],
    'resnet101_v1d': [64, 128, 256, 512],
    'resnet152_v1d': [64, 128, 256, 512],
    'resnet50_v1s': [64, 128, 256, 512]
}


def create_resotre_op(scope, mxnet_weights_path):

    mxnetName_array_dict = read_mxnet_weights(mxnet_weights_path, show=False)

    tf_mxnet_map, mxnet_tf_map = \
        weights_map.get_map(scope=scope,
                            bottleneck_nums=BottleNeck_NUM_DICT[scope], show_mxnettf=False, show_tfmxnet=False)

    tf_model_vars = slim.get_model_variables(scope)

    # # check name and var
    check_mxnet_names(mxnet_tf_map, mxnetName_array_dict=mxnetName_array_dict)
    check_tf_vars(tf_mxnet_map, mxnetName_array_dict, tf_model_vars, scope=scope)
    # #

    assign_ops = []

    for var in tf_model_vars:
        name = var.op.name.split('%s/' % scope)[1]
        new_val = tf.constant(mxnetName_array_dict[tf_mxnet_map[name]])
        sub_assign_op = tf.assign(var, value=new_val)

        assign_ops.append(sub_assign_op)

    assign_op = tf.group(*assign_ops)

    return assign_op


def build_resnet(img_batch=None, scope='resnet50_v1d', is_training=True, freeze_norm=False, num_cls=1000):
    if img_batch is None:
        np.random.seed(30)
        img_batch = np.random.rand(1, 224, 224, 3)  # H, W, C
        img_batch = tf.constant(img_batch, dtype=tf.float32)

    print("Please Ensure the img is in NHWC")

    if scope.endswith('b'):
        get_resnet_fn = get_resnet_v1_b
    elif scope.endswith('d'):
        get_resnet_fn = get_resnet_v1_d
    elif scope.endswith('s'):
        get_resnet_fn = get_resnet_v1_s

    logits = get_resnet_fn(input_x=img_batch, scope=scope,
                           bottleneck_nums=BottleNeck_NUM_DICT[scope],
                           base_channels=BASE_CHANNELS_DICT[scope],
                           is_training=is_training, freeze_norm=freeze_norm, num_cls=num_cls)

    return logits


if __name__ == "__main__":
    build_resnet()
    create_resotre_op()
