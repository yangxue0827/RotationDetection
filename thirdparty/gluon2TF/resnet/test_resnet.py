# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from mxnet import nd, image
import numpy as np
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.imagenet import transform_eval
from resnet import build_resnet, create_resotre_op
from resnet_utils import DEBUG
import os


MODEL_NAME = 'resnet50_v1d'
Mxnet_Weights_PATH = '../mxnet_weights/resnet50_v1d-117a384e.params'


def mxnet_process_img(path):
    # Load Images
    img = image.imread(path)

    # Transform
    img = transform_eval(img)
    img_arr = img.asnumpy()
    if len(img_arr) == 3:
        img_arr = np.expand_dims(img_arr, axis=0)
    img_tf_tensor = tf.constant(img_arr)

    # np.random.seed(30)
    # img = nd.array(np.random.randn(1, 3, 600, 800))
    # img_tf_tensor = tf.constant(img.asnumpy())
    img_tf_tensor = tf.transpose(img_tf_tensor, [0, 2, 3, 1])
    return img, img_tf_tensor


def mxnet_infer(img):

    model_name = MODEL_NAME
    net = get_model(model_name, pretrained=False)
    net.load_parameters(Mxnet_Weights_PATH)
    pred = net(img)

    # print(pred.shape, pred.dtype)
    pred = pred.asnumpy()
    return pred


def tf_infer(img, save_ckpt=True, restore_from_tfckpt=False, ckpt_path=None):

    pred_tensor = build_resnet(img_batch=img, scope=MODEL_NAME,
                               is_training=False, freeze_norm=True, num_cls=1000)

    if restore_from_tfckpt:
        print("restore weights from tf_CKPT")
        assert not ckpt_path is None, "ckpt_path is None, Erro"
        restore_op = tf.train.Saver()
    else:
        print('restore weights from MxnetWeights')
        restore_op = create_resotre_op(MODEL_NAME, Mxnet_Weights_PATH)

    if DEBUG:
        from resnet_utils import debug_dict
        print(debug_dict)
        assert len(debug_dict) >=3, "debug_dict size erro, len is :{}".format(len(debug_dict))

    if save_ckpt:
        save_dir = '../tf_ckpts'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        saver = tf.train.Saver(max_to_keep=30)
        save_ckpt = os.path.join(save_dir, '%s.ckpt' % MODEL_NAME)

    with tf.Session() as sess:
        if restore_from_tfckpt:
            restore_op.restore(sess, ckpt_path)
        else:
            sess.run(restore_op)
        if DEBUG:
            name_val = {}
            for name in debug_dict.keys():
                name_val[name] = sess.run(debug_dict[name])
        pred = sess.run(pred_tensor)
        if save_ckpt:
            saver.save(sess, save_ckpt)

    return pred


def cal_erro(img_path, use_tf_ckpt=False, ckpt_path=None, save_ckpt=False):

    mxnet_img, tf_img = mxnet_process_img(img_path)

    mxnet_pred = mxnet_infer(mxnet_img)

    mxnet_pred = np.squeeze(mxnet_pred, axis=0)
    tf_pred = tf_infer(tf_img, restore_from_tfckpt=use_tf_ckpt, ckpt_path=ckpt_path, save_ckpt=save_ckpt)
    tf_pred = np.squeeze(tf_pred, axis=0)
    assert mxnet_pred.shape == tf_pred.shape, "mxnet_pred shape Do Not equal with tf_pred shape"

    argmax_mxnet = np.argmax(mxnet_pred)
    argmax_tf = np.argmax(tf_pred)

    erro = np.linalg.norm(tf_pred-mxnet_pred)
    for i, (m, t) in enumerate(zip(mxnet_pred, tf_pred)):
        if i == 5:
            break
        print("mxnet|tf==>>{} | {} ".format(m, t))

    print('total_erro-->', erro)
    print('erro_rate-->', erro/np.linalg.norm(mxnet_pred))
    print("argmax_mxnet: {} || tf_argmx: {}".format(argmax_mxnet, argmax_tf))


if __name__ == '__main__':

    cal_erro(img_path='../demo_img/person.jpg',
             use_tf_ckpt=False,
             ckpt_path=None,
             save_ckpt=True)
    # cal_erro(img_path='../demo_img/person.jpg',
    #          use_tf_ckpt=True,
    #          ckpt_path='../tf_ckpts/%s.ckpt' % MODEL_NAME,
    #          save_ckpt=False)
    print(20*"++")
