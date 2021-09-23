# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
import numpy as np
import sys
import os
sys.path.append('../')
from libs.configs import cfgs


class SWA(object):

    """
    SWA Object Detection
    https://arxiv.org/pdf/2012.12645.pdf
    """

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.weight_paths = self.get_all_weights_path()

    def get_all_weights_path(self):
        with open(os.path.join('../output/trained_weights', self.cfgs.VERSION, 'checkpoint'), 'r') as fr:
            lines = fr.readlines()[1:]
        return [line.split(' ')[-1][1:-2] for line in lines]

    def average_weights(self, weight_paths):
        print(weight_paths)
        average_weights = {}
        for wp in weight_paths:
            print(wp)
            model_reader = pywrap_tensorflow.NewCheckpointReader(wp)
            var_dict = model_reader.get_variable_to_shape_map()
            for key in var_dict:
                if key not in average_weights.keys():
                    average_weights[key] = model_reader.get_tensor(key)
                else:
                    average_weights[key] += model_reader.get_tensor(key)
        for key in average_weights.keys():
            average_weights[key] /= len(weight_paths)
        return average_weights

    def save_swa_weight(self):
        # img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
        # img_batch = tf.cast(img_plac, tf.float32)
        #
        # if self.cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        #     img_batch = (img_batch / 255 - tf.constant(self.cfgs.PIXEL_MEAN_)) / tf.constant(self.cfgs.PIXEL_STD)
        # else:
        #     img_batch = img_batch - tf.constant(self.cfgs.PIXEL_MEAN)
        #
        # img_batch = tf.expand_dims(img_batch, axis=0)
        #
        # _ = det_net.build_whole_detection_network(input_img_batch=img_batch)

        average_weights = self.average_weights(self.weight_paths)
        weights = {}
        for k in average_weights.keys():
            weights[k] = tf.get_variable(name=k, initializer=average_weights[k])

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        # average_weights = self.average_weights(weight_paths)

        # model_variables = slim.get_model_variables()
        # for i in range(len(model_variables)):
        #     model_variables[i] = model_variables[i].assign(average_weights[model_variables[i].name.split(':')[0]])

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init_op)
            # for i in range(len(model_variables)):
            #     model_variables[i].op.run()
            saver.save(sess, "../output/trained_weights/{}/swa_{}.ckpt".format(self.cfgs.VERSION, len(self.weight_paths)))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    swa = SWA(cfgs)
    swa.save_swa_weight()
