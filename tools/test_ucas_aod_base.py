# -*- coding:utf-8 -*-
# Author: Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>
#
# License: Apache-2.0 license

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import time
import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm
sys.path.append("../")

from utils import tools
from libs.label_name_dict.label_dict import LabelMap
from libs.utils.draw_box_in_img import DrawBox
from libs.utils.coordinate_convert import forward_convert, backward_convert
from libs.utils import nms_rotate
from libs.utils.rotate_polygon_nms import rotate_gpu_nms
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test HRSC2016')
    parser.add_argument('--img_dir', dest='img_dir',
                        help='images path',
                        default='/data/dataset_share/UCAS-AOD/VOCdevkit_test/JPEGImages', type=str)
    parser.add_argument('--image_ext', dest='image_ext',
                        help='image format',
                        default='.png', type=str)
    parser.add_argument('--test_annotation_path', dest='test_annotation_path',
                        help='test annotate path',
                        default='/data/dataset_share/UCAS-AOD/VOCdevkit_test/Annotations', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)
    parser.add_argument('--draw_imgs', '-s', default=False,
                        action='store_true')
    parser.add_argument('--multi_scale', '-ms', default=False,
                        action='store_true')
    parser.add_argument('--cpu_nms', '-cn', default=False,
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class TestUCASAOD(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.args = parse_args()
        label_map = LabelMap(cfgs)
        self.name_label_map, self.label_name_map = label_map.name2label(), label_map.label2name()

    def eval_with_plac(self, img_dir, det_net, image_ext):

        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        # 1. preprocess img
        img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
        img_batch = tf.cast(img_plac, tf.float32)

        pretrain_zoo = PretrainModelZoo()
        if self.cfgs.NET_NAME in pretrain_zoo.pth_zoo or self.cfgs.NET_NAME in pretrain_zoo.mxnet_zoo:
            img_batch = (img_batch / 255 - tf.constant(self.cfgs.PIXEL_MEAN_)) / tf.constant(self.cfgs.PIXEL_STD)
        else:
            img_batch = img_batch - tf.constant(self.cfgs.PIXEL_MEAN)

        img_batch = tf.expand_dims(img_batch, axis=0)

        detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
            input_img_batch=img_batch)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer, restore_ckpt = det_net.get_restorer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            all_boxes_r = []
            imgs = os.listdir(img_dir)
            pbar = tqdm(imgs)
            for a_img_name in pbar:
                a_img_name = a_img_name.split(image_ext)[0]

                raw_img = cv2.imread(os.path.join(img_dir,
                                                  a_img_name + image_ext))
                raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

                det_boxes_r_all, det_scores_r_all, det_category_r_all = [], [], []

                img_short_side_len_list = self.cfgs.IMG_SHORT_SIDE_LEN if isinstance(self.cfgs.IMG_SHORT_SIDE_LEN, list) else [
                    self.cfgs.IMG_SHORT_SIDE_LEN]
                img_short_side_len_list = [img_short_side_len_list[0]] if not self.args.multi_scale else img_short_side_len_list

                for short_size in img_short_side_len_list:
                    max_len = self.cfgs.IMG_MAX_LENGTH
                    if raw_h < raw_w:
                        new_h, new_w = short_size, min(int(short_size * float(raw_w) / raw_h), max_len)
                    else:
                        new_h, new_w = min(int(short_size * float(raw_h) / raw_w), max_len), short_size
                    img_resize = cv2.resize(raw_img, (new_w, new_h))

                    resized_img, detected_boxes, detected_scores, detected_categories = \
                        sess.run(
                            [img_batch, detection_boxes, detection_scores, detection_category],
                            feed_dict={img_plac: img_resize[:, :, ::-1]}
                        )

                    if detected_boxes.shape[0] == 0:
                        continue
                    resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                    detected_boxes = forward_convert(detected_boxes, False)
                    detected_boxes[:, 0::2] *= (raw_w / resized_w)
                    detected_boxes[:, 1::2] *= (raw_h / resized_h)

                    det_boxes_r_all.extend(detected_boxes)
                    det_scores_r_all.extend(detected_scores)
                    det_category_r_all.extend(detected_categories)
                det_boxes_r_all = np.array(det_boxes_r_all)
                det_scores_r_all = np.array(det_scores_r_all)
                det_category_r_all = np.array(det_category_r_all)

                box_res_rotate_ = []
                label_res_rotate_ = []
                score_res_rotate_ = []

                if det_scores_r_all.shape[0] != 0:
                    for sub_class in range(1, self.cfgs.CLASS_NUM + 1):
                        index = np.where(det_category_r_all == sub_class)[0]
                        if len(index) == 0:
                            continue
                        tmp_boxes_r = det_boxes_r_all[index]
                        tmp_label_r = det_category_r_all[index]
                        tmp_score_r = det_scores_r_all[index]

                        if self.args.multi_scale:
                            tmp_boxes_r_ = backward_convert(tmp_boxes_r, False)

                            # try:
                            #     inx = nms_rotate.nms_rotate_cpu(boxes=np.array(tmp_boxes_r_),
                            #                                     scores=np.array(tmp_score_r),
                            #                                     iou_threshold=self.cfgs.NMS_IOU_THRESHOLD,
                            #                                     max_output_size=5000)
                            # except:
                            tmp_boxes_r_ = np.array(tmp_boxes_r_)
                            tmp = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                            tmp[:, 0:-1] = tmp_boxes_r_
                            tmp[:, -1] = np.array(tmp_score_r)
                            # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
                            jitter = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                            jitter[:, 0] += np.random.rand(tmp_boxes_r_.shape[0], ) / 1000
                            inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                                 float(self.cfgs.NMS_IOU_THRESHOLD), 0)
                        else:
                            inx = np.arange(0, tmp_score_r.shape[0])

                        box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
                        score_res_rotate_.extend(np.array(tmp_score_r)[inx])
                        label_res_rotate_.extend(np.array(tmp_label_r)[inx])

                if len(box_res_rotate_) == 0:
                    all_boxes_r.append(np.array([]))
                    continue

                det_boxes_r_ = np.array(box_res_rotate_)
                det_scores_r_ = np.array(score_res_rotate_)
                det_category_r_ = np.array(label_res_rotate_)

                if self.args.draw_imgs:
                    detected_indices = det_scores_r_ >= self.cfgs.VIS_SCORE
                    detected_scores = det_scores_r_[detected_indices]
                    detected_boxes = det_boxes_r_[detected_indices]
                    detected_categories = det_category_r_[detected_indices]

                    detected_boxes = backward_convert(detected_boxes, False)

                    drawer = DrawBox(self.cfgs)

                    det_detections_r = drawer.draw_boxes_with_label_and_scores(raw_img[:, :, ::-1],
                                                                               boxes=detected_boxes,
                                                                               labels=detected_categories,
                                                                               scores=detected_scores,
                                                                               method=1,
                                                                               in_graph=True)

                    save_dir = os.path.join('test_hrsc', self.cfgs.VERSION, 'hrsc2016_img_vis')
                    tools.makedirs(save_dir)

                    cv2.imwrite(save_dir + '/{}.jpg'.format(a_img_name),
                                det_detections_r[:, :, ::-1])

                det_boxes_r_ = backward_convert(det_boxes_r_, False)

                x_c, y_c, w, h, theta = det_boxes_r_[:, 0], det_boxes_r_[:, 1], det_boxes_r_[:, 2], \
                                        det_boxes_r_[:, 3], det_boxes_r_[:, 4]

                boxes_r = np.transpose(np.stack([x_c, y_c, w, h, theta]))
                dets_r = np.hstack((det_category_r_.reshape(-1, 1),
                                    det_scores_r_.reshape(-1, 1),
                                    boxes_r))
                all_boxes_r.append(dets_r)

                pbar.set_description("Eval image %s" % a_img_name)

            # fw1 = open(cfgs.VERSION + '_detections_r.pkl', 'wb')
            # pickle.dump(all_boxes_r, fw1)
            return all_boxes_r


