# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from libs.utils.draw_box_in_img import DrawBox


class DrawBoxTensor(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.drawer = DrawBox(cfgs)

    def only_draw_boxes(self, img_batch, boxes, method, head=None, is_csl=False):

        boxes = tf.stop_gradient(boxes)
        img_tensor = tf.squeeze(img_batch, 0)
        img_tensor = tf.cast(img_tensor, tf.float32)
        labels = tf.ones(shape=(tf.shape(boxes)[0], ), dtype=tf.int32) * self.drawer.ONLY_DRAW_BOXES
        scores = tf.zeros_like(labels, dtype=tf.float32)

        if head is None:
            head = tf.ones_like(scores) * -1

        img_tensor_with_boxes = tf.py_func(self.drawer.draw_boxes_with_label_and_scores,
                                           inp=[img_tensor, boxes, labels, scores, method, head, is_csl],
                                           Tout=tf.uint8)
        img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))  # [batch_size, h, w, c]

        return img_tensor_with_boxes

    def draw_boxes_with_scores(self, img_batch, boxes, scores, method, head, is_csl=False):

        if head is None:
            head = tf.ones_like(scores) * -1

        boxes = tf.stop_gradient(boxes)
        scores = tf.stop_gradient(scores)

        img_tensor = tf.squeeze(img_batch, 0)
        img_tensor = tf.cast(img_tensor, tf.float32)
        labels = tf.ones(shape=(tf.shape(boxes)[0],), dtype=tf.int32) * self.drawer.ONLY_DRAW_BOXES_WITH_SCORES
        img_tensor_with_boxes = tf.py_func(self.drawer.draw_boxes_with_label_and_scores,
                                           inp=[img_tensor, boxes, labels, scores, method, head, is_csl],
                                           Tout=[tf.uint8])
        img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
        return img_tensor_with_boxes

    def draw_boxes_with_categories(self, img_batch, boxes, labels, method, head=None, is_csl=False):

        if head is None:
            head = tf.ones_like(labels) * -1

        boxes = tf.stop_gradient(boxes)

        img_tensor = tf.squeeze(img_batch, 0)
        img_tensor = tf.cast(img_tensor, tf.float32)
        scores = tf.ones(shape=(tf.shape(boxes)[0],), dtype=tf.float32)

        img_tensor_with_boxes = tf.py_func(self.drawer.draw_boxes_with_label_and_scores,
                                           inp=[img_tensor, boxes, labels, scores, method, head, is_csl],
                                           Tout=[tf.uint8])
        img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
        return img_tensor_with_boxes

    def draw_boxes_with_categories_and_scores(self, img_batch, boxes, labels, scores, method, head=None, is_csl=False):

        if head is None:
            head = tf.ones_like(labels) * -1

        boxes = tf.stop_gradient(boxes)
        scores = tf.stop_gradient(scores)

        img_tensor = tf.squeeze(img_batch, 0)
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor_with_boxes = tf.py_func(self.drawer.draw_boxes_with_label_and_scores,
                                           inp=[img_tensor, boxes, labels, scores, method, head, is_csl],
                                           Tout=[tf.uint8])
        img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
        return img_tensor_with_boxes


