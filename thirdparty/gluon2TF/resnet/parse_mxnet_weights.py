# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import mxnet.ndarray as nd
import numpy as np


def read_mxnet_weights(path, show=False):

    # assert os.path.exists(path), "path erro: {}".format(path)

    name_MxnetArray_dict = nd.load(path)

    name_array_dict = {}
    for name in sorted(name_MxnetArray_dict.keys()):
        mxnet_array = name_MxnetArray_dict[name]
        array = mxnet_array.asnumpy()

        if show:
            print ("name: {} || shape: {} || dtype: {}".format(name, array.shape, array.dtype))

        if name.endswith("weight"):
            if name.endswith("fc.weight"):
                array = np.transpose(array, [1, 0])
            else:
                array = np.transpose(array, [2, 3, 1, 0])
            # (out_channel, in_channel, k, k)(mxnet) --> (k, k, in_channel, out_channel)(tf)
            # (32, 3, 3, 3)-->(3, 3, 3, 32)
        name_array_dict[name] = array

    return name_array_dict


def check_mxnet_names(mxnet_tf_map, mxnetName_array_dict):

    for key1, key2 in zip(sorted(mxnet_tf_map.keys()), sorted(mxnetName_array_dict.keys())):
        assert key1 == key2, "key in mxnet_array_dict and mxnet_tf_map do not equal, details are :\n" \
                             "key1 in mxnet_tf_map: {}\n"\
                             "key2 in mxnet_array dict: {}".format(key1, key2)
    if len(mxnetName_array_dict) == len(mxnet_tf_map):
        print("all mxnet names are mapped")


def check_tf_vars(tf_mxnet_map, mxnetName_array_dict, tf_model_vars, scope='resnet50_v1_d'):

    tf_nake_names = sorted([var.op.name.split("%s/" % scope)[1] for var in tf_model_vars])
    # check_name
    for tf_name, name2 in zip(tf_nake_names, sorted(tf_mxnet_map.keys())):
        assert tf_name == name2, "key in tf_model_vars and tf_mxnet_map do not equal, details are :\n" \
                                 "tf_name in tf_model_vars: {}\n" \
                                 "name2 in tf_mxnet_maps: {}".format(tf_name, name2)
    print("all tf_model_var can find matched name in tf_mxnet_map")

    # check shape
    for var in tf_model_vars:
        name = var.op.name.split("%s/"%scope)[1]
        array = mxnetName_array_dict[tf_mxnet_map[name]]

        assert var.shape == array.shape,  "var in tf_model_vars and mxnet_arrays shape do not equal, details are :\n" \
                                          "tf_var in tf_model_vars: {}\n" \
                                          "name in tf_mxnet_maps: {}, shape is : {}".format(var, tf_mxnet_map[name],
                                                                                            array.shape)
    print("All tf_model_var shapes match the shape of arrays in mxnet_array_dict...")


