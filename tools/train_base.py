# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import sys
import numpy as np
import time
sys.path.append("../")

from dataloader.dataset.read_tfrecord import ReadTFRecord
from libs.utils.show_box_in_tensor import DrawBoxTensor
from utils import tools


class Train(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.reader = ReadTFRecord(cfgs)
        self.drawer = DrawBoxTensor(cfgs)

    def stats_graph(self, graph):
        flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def sum_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        sum_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_sum(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            sum_grads.append(grad_and_var)
        return sum_grads

    def warmup_lr(self, init_lr, global_step, warmup_step, num_gpu):
        def warmup(end_lr, global_step, warmup_step):
            start_lr = end_lr * 0.1
            global_step = tf.cast(global_step, tf.float32)
            return start_lr + (end_lr - start_lr) * global_step / warmup_step

        def decay(start_lr, global_step, num_gpu):
            lr = tf.train.piecewise_constant(global_step,
                                             boundaries=[np.int64(self.cfgs.DECAY_STEP[0] // num_gpu),
                                                         np.int64(self.cfgs.DECAY_STEP[1] // num_gpu),
                                                         np.int64(self.cfgs.DECAY_STEP[2] // num_gpu)],
                                             values=[start_lr, start_lr / 10., start_lr / 100., start_lr / 1000.])
            return lr

        return tf.cond(tf.less_equal(global_step, warmup_step),
                       true_fn=lambda: warmup(init_lr, global_step, warmup_step),
                       false_fn=lambda: decay(init_lr, global_step, num_gpu))

    def log_printer(self, deter, optimizer, global_step, tower_grads, total_loss_dict, num_gpu, graph):
        for k in total_loss_dict.keys():
            tf.summary.scalar('{}/{}'.format(k.split('_')[0], k), total_loss_dict[k])

        if len(tower_grads) > 1:
            grads = self.sum_gradients(tower_grads)
        else:
            grads = tower_grads[0]

        if self.cfgs.MUTILPY_BIAS_GRADIENT is not None:
            final_gvs = []
            with tf.variable_scope('Gradient_Mult'):
                for grad, var in grads:
                    scale = 1.
                    if '/biases:' in var.name:
                        scale *= self.cfgs.MUTILPY_BIAS_GRADIENT
                    if 'conv_new' in var.name:
                        scale *= 3.
                    if not np.allclose(scale, 1.0):
                        grad = tf.multiply(grad, scale)

                    final_gvs.append((grad, var))
            apply_gradient_op = optimizer.apply_gradients(final_gvs, global_step=global_step)
        else:
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        train_op = tf.group(apply_gradient_op, variables_averages_op)
        # train_op = optimizer.apply_gradients(final_gvs, global_step=global_step)
        summary_op = tf.summary.merge_all()

        restorer, restore_ckpt = deter.get_restorer()
        saver = tf.train.Saver(max_to_keep=5)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        tfconfig = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        tfconfig.gpu_options.allow_growth = True
        with tf.Session(config=tfconfig) as sess:
            sess.run(init_op)

            # sess.run(tf.initialize_all_variables())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            summary_path = os.path.join(self.cfgs.SUMMARY_PATH, self.cfgs.VERSION)
            tools.makedirs(summary_path)
            summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            self.stats_graph(graph)

            for step in range(self.cfgs.MAX_ITERATION // num_gpu):
                training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

                if step % self.cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % self.cfgs.SMRY_ITER != 0:
                    _, global_stepnp = sess.run([train_op, global_step])

                else:
                    if step % self.cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % self.cfgs.SMRY_ITER != 0:
                        start = time.time()

                        _, global_stepnp, total_loss_dict_ = \
                            sess.run([train_op, global_step, total_loss_dict])

                        end = time.time()

                        print('***'*20)
                        print("""%s: global_step:%d  current_step:%d"""
                              % (training_time, (global_stepnp-1)*num_gpu, step*num_gpu))
                        print("""per_cost_time:%.3fs"""
                              % ((end - start) / num_gpu))
                        loss_str = ''
                        for k in total_loss_dict_.keys():
                            loss_str += '%s:%.3f\n' % (k, total_loss_dict_[k])
                        print(loss_str)

                        if np.isnan(total_loss_dict_['total_losses']):
                            sys.exit(0)

                    else:
                        if step % self.cfgs.SMRY_ITER == 0:
                            _, global_stepnp, summary_str = sess.run([train_op, global_step, summary_op])
                            summary_writer.add_summary(summary_str, (global_stepnp-1)*num_gpu)
                            summary_writer.flush()

                if (step > 0 and step % (self.cfgs.SAVE_WEIGHTS_INTE // num_gpu) == 0) or (step >= self.cfgs.MAX_ITERATION // num_gpu - 1):

                    save_dir = os.path.join(self.cfgs.TRAINED_CKPT, self.cfgs.VERSION)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)

                    save_ckpt = os.path.join(save_dir, '{}_'.format(self.cfgs.DATASET_NAME) +
                                             str((global_stepnp-1)*num_gpu) + 'model.ckpt')
                    saver.save(sess, save_ckpt)
                    print(' weights had been saved')

            coord.request_stop()
            coord.join(threads)












