import cv2
import tensorflow as tf
import os
import sys

sys.path.append('../../..')
from libs.networks.efficientnet import efficientnet_builder

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def restore_model(sess, ckpt_dir):
    """Restore variables from checkpoint dir."""
    checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
    for v in tf.global_variables():
        if 'moving_mean' in v.name or 'moving_variance' in v.name:
            ema_vars.append(v)
    ema_vars = list(set(ema_vars))
    var_dict = ema.variables_to_restore(ema_vars)
    saver = tf.train.Saver(max_to_keep=1)
    saver.restore(sess, checkpoint)


images = cv2.imread('/data/yangxue/code/R3Det_Tensorflow/libs/networks/efficientnet/panda.jpg')
images = cv2.resize(images, (112, 112))
images = tf.expand_dims(tf.constant(images, tf.float32), axis=0)
features, endpoints = efficientnet_builder.build_model_base(images, 'efficientnet-b0', training=True)
print(endpoints.keys())

init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

tfconfig = tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=False)
tfconfig.gpu_options.allow_growth = True
with tf.Session(config=tfconfig) as sess:
    sess.run(init_op)
    restore_model(sess, '/data/yangxue/code/R3Det_Tensorflow/libs/networks/efficientnet/efficientnet-b0')
    features_, endpoints_ = sess.run([features, endpoints])
    print(endpoints['reduction_1'])
    print(endpoints['reduction_2'])
    print(endpoints['reduction_3'])
    print(endpoints['reduction_4'])
    print(endpoints['reduction_5'])