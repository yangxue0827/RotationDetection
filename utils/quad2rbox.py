import numpy as np
import math
import tensorflow as tf
import sys
sys.path.append('../')
from libs.utils.coordinate_convert import backward_convert


def dist(p1, p2):
    return np.linalg.norm(p1 - p2, axis=1)


def quad2rbox(boxes):
    # input: ordered points (bs, 8) 
    nB = len(boxes)
    points = boxes.reshape(-1, 4, 2)
    cxs = points[:, :, 0].sum(1)[:, np.newaxis] / 4
    cys = points[:, :, 1].sum(1)[:, np.newaxis] / 4
    _ws = dist(points[:, 0], points[:, 1])[:, np.newaxis]
    _hs = dist(points[:, 1], points[:, 2])[:, np.newaxis]
    # adjust theta
    _thetas = np.arctan2(-(points[:, 1, 0] - points[:, 0, 0]), points[:, 1, 1] - points[:, 0, 1])[:, np.newaxis]
    odd = (np.mod((_thetas // (-np.pi * 0.5)), 2) == 0)
    ws = np.where(odd, _hs, _ws)
    hs = np.where(odd, _ws, _hs)
    thetas = np.mod(_thetas, -np.pi * 0.5)
    rboxes = np.concatenate([cxs, cys, ws, hs, thetas / np.pi * 180.], 1)

    return rboxes


def dist_tf(p1, p2):
    return tf.linalg.norm(p1 - p2, axis=-1)


def quad2rbox_tf(boxes):
    points = tf.reshape(boxes, [-1, 4, 2])
    cxs = tf.expand_dims(tf.reduce_sum(points[:, :, 0], axis=1), axis=1) / 4.
    cys = tf.expand_dims(tf.reduce_sum(points[:, :, 1], axis=1), axis=1) / 4.
    _ws = tf.expand_dims(dist_tf(points[:, 0], points[:, 1]), axis=1)
    _hs = tf.expand_dims(dist_tf(points[:, 1], points[:, 2]), axis=1)
    _thetas = tf.expand_dims(tf.atan2(-(points[:, 1, 0]-points[:, 0, 0]), points[:, 1, 1]-points[:, 0, 1]), axis=1)
    odd = tf.equal(tf.mod((_thetas // (-np.pi * 0.5)), 2), 0)
    ws = tf.where(odd, _hs, _ws)
    hs = tf.where(odd, _ws, _hs)
    thetas = tf.mod(_thetas, -np.pi * 0.5)
    rboxes = tf.concat([cxs, cys, ws, hs, thetas / np.pi * 180.], axis=1)
    return rboxes


if __name__ == "__main__":
    quad = np.array([[278, 418, 308, 331, 761, 581, 691, 668],
                     [758, 418, 348, 331, 241, 581, 591, 668],
                     [624, 112, 490, 93, 496, 50, 630, 68],
                     [10, 0, 30, 20, 20, 30, 0, 10]], np.float32)
    box = quad2rbox(quad)
    print(box)
    box_tf = quad2rbox_tf(quad)
    with tf.Session() as sess:
        box_tf_ = sess.run(box_tf)
        print(box_tf_)

        print(backward_convert(quad, False))


