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
import cv2
import numpy as np
import math
from tqdm import tqdm
import argparse
from multiprocessing import Queue, Process

from utils import tools
from libs.label_name_dict.label_dict import LabelMap
from libs.utils.draw_box_in_img import DrawBox
from libs.utils.coordinate_convert import forward_convert, backward_convert
from libs.utils import nms_rotate
from libs.utils.rotate_polygon_nms import rotate_gpu_nms
from utils.order_points import sort_corners
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo


def parse_args():

    parser = argparse.ArgumentParser('Test MSRA-TD500')

    parser.add_argument('--test_dir', dest='test_dir',
                        help='evaluate imgs dir ',
                        default='/data/dataset_share/MSRA-TD500/test', type=str)
    parser.add_argument('--gpus', dest='gpus',
                        help='gpu id',
                        default='0,1,2,3,4,5,6,7', type=str)
    parser.add_argument('--num_imgs', dest='num_imgs',
                        help='test image number',
                        default=np.inf, type=int)
    parser.add_argument('--show_box', '-s', default=False,
                        action='store_true')
    parser.add_argument('--flip_img', '-f', default=False,
                        action='store_true')
    parser.add_argument('--multi_scale', '-ms', default=False,
                        action='store_true')
    parser.add_argument('--cpu_nms', '-cn', default=False,
                        action='store_true')
    args = parser.parse_args()
    return args


class TestIMSRATD500(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.args = parse_args()
        label_map = LabelMap(cfgs)
        self.name_label_map, self.label_name_map = label_map.name2label(), label_map.label2name()

    def worker(self, gpu_id, images, det_net, result_queue):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
            input_img_batch=img_batch,
            gtboxes_batch_h=None,
            gtboxes_batch_r=None,
            gpu_id=0)

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
                print('restore model %d ...' % gpu_id)
            for a_img in images:
                raw_img = cv2.imread(a_img)
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

                    detected_indices = detected_scores >= self.cfgs.VIS_SCORE
                    detected_scores = detected_scores[detected_indices]
                    detected_boxes = detected_boxes[detected_indices]
                    detected_categories = detected_categories[detected_indices]

                    if detected_boxes.shape[0] == 0:
                        continue
                    resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                    detected_boxes = forward_convert(detected_boxes, False)
                    detected_boxes[:, 0::2] *= (raw_w / resized_w)
                    detected_boxes[:, 1::2] *= (raw_h / resized_h)

                    det_boxes_r_all.extend(detected_boxes)
                    det_scores_r_all.extend(detected_scores)
                    det_category_r_all.extend(detected_categories)

                    if self.args.flip_img:
                        detected_boxes, detected_scores, detected_categories = \
                            sess.run(
                                [detection_boxes, detection_scores, detection_category],
                                feed_dict={img_plac: cv2.flip(img_resize, flipCode=1)[:, :, ::-1]}
                            )
                        detected_indices = detected_scores >= self.cfgs.VIS_SCORE
                        detected_scores = detected_scores[detected_indices]
                        detected_boxes = detected_boxes[detected_indices]
                        detected_categories = detected_categories[detected_indices]

                        if detected_boxes.shape[0] == 0:
                            continue
                        resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                        detected_boxes = forward_convert(detected_boxes, False)
                        detected_boxes[:, 0::2] *= (raw_w / resized_w)
                        detected_boxes[:, 0::2] = (raw_w - detected_boxes[:, 0::2])
                        detected_boxes[:, 1::2] *= (raw_h / resized_h)

                        det_boxes_r_all.extend(sort_corners(detected_boxes))
                        det_scores_r_all.extend(detected_scores)
                        det_category_r_all.extend(detected_categories)

                        detected_boxes, detected_scores, detected_categories = \
                            sess.run(
                                [detection_boxes, detection_scores, detection_category],
                                feed_dict={img_plac: cv2.flip(img_resize, flipCode=0)[:, :, ::-1]}
                            )
                        detected_indices = detected_scores >= self.cfgs.VIS_SCORE
                        detected_scores = detected_scores[detected_indices]
                        detected_boxes = detected_boxes[detected_indices]
                        detected_categories = detected_categories[detected_indices]

                        if detected_boxes.shape[0] == 0:
                            continue
                        resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                        detected_boxes = forward_convert(detected_boxes, False)
                        detected_boxes[:, 0::2] *= (raw_w / resized_w)
                        detected_boxes[:, 1::2] *= (raw_h / resized_h)
                        detected_boxes[:, 1::2] = (raw_h - detected_boxes[:, 1::2])
                        det_boxes_r_all.extend(sort_corners(detected_boxes))
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

                            # cpu nms better than gpu nms (default)
                            if self.args.cpu_nms:
                                try:
                                    inx = nms_rotate.nms_rotate_cpu(boxes=np.array(tmp_boxes_r_),
                                                                    scores=np.array(tmp_score_r),
                                                                    iou_threshold=self.cfgs.NMS_IOU_THRESHOLD,
                                                                    max_output_size=5000)

                                except:
                                    tmp_boxes_r_ = np.array(tmp_boxes_r_)
                                    tmp = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                                    tmp[:, 0:-1] = tmp_boxes_r_
                                    tmp[:, -1] = np.array(tmp_score_r)
                                    # Note: the IoU of two same rectangles is 0
                                    jitter = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                                    jitter[:, 0] += np.random.rand(tmp_boxes_r_.shape[0], ) / 1000
                                    inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                                         float(self.cfgs.NMS_IOU_THRESHOLD), 0)
                            else:
                                tmp_boxes_r_ = np.array(tmp_boxes_r_)
                                tmp = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                                tmp[:, 0:-1] = tmp_boxes_r_
                                tmp[:, -1] = np.array(tmp_score_r)
                                # Note: the IoU of two same rectangles is 0
                                jitter = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                                jitter[:, 0] += np.random.rand(tmp_boxes_r_.shape[0], ) / 1000
                                inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                                     float(self.cfgs.NMS_IOU_THRESHOLD), 0)
                        else:
                            inx = np.arange(0, tmp_score_r.shape[0])

                        box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
                        score_res_rotate_.extend(np.array(tmp_score_r)[inx])
                        label_res_rotate_.extend(np.array(tmp_label_r)[inx])

                box_res_rotate_ = np.array(box_res_rotate_)
                score_res_rotate_ = np.array(score_res_rotate_)
                label_res_rotate_ = np.array(label_res_rotate_)

                result_dict = {'scales': [1, 1], 'boxes': box_res_rotate_,
                               'scores': score_res_rotate_, 'labels': label_res_rotate_,
                               'image_id': a_img}
                result_queue.put_nowait(result_dict)

    def test_msra_td500(self, det_net, real_test_img_list, txt_name):

        save_path = os.path.join('./test_msra_td500', self.cfgs.VERSION)
        tools.makedirs(save_path)

        nr_records = len(real_test_img_list)
        pbar = tqdm(total=nr_records)
        gpu_num = len(self.args.gpus.strip().split(','))

        nr_image = math.ceil(nr_records / gpu_num)
        result_queue = Queue(500)
        procs = []

        for i, gpu_id in enumerate(self.args.gpus.strip().split(',')):
            start = i * nr_image
            end = min(start + nr_image, nr_records)
            split_records = real_test_img_list[start:end]
            proc = Process(target=self.worker, args=(int(gpu_id), split_records, det_net, result_queue))
            print('process:%d, start:%d, end:%d' % (i, start, end))
            proc.start()
            procs.append(proc)

        for i in range(nr_records):
            res = result_queue.get()
            tools.makedirs(os.path.join(save_path, 'msra_td500_res'))
            if res['boxes'].shape[0] == 0:
                fw_txt_dt = open(os.path.join(save_path, 'msra_td500_res', 'res_{}.txt'.format(res['image_id'].split('/')[-1].split('.')[0]).replace('IMG', 'img')),
                                 'w')
                fw_txt_dt.close()
                pbar.update(1)

                fw = open(txt_name, 'a+')
                fw.write('{}\n'.format(res['image_id'].split('/')[-1]))
                fw.close()
                continue
            x1, y1, x2, y2, x3, y3, x4, y4 = res['boxes'][:, 0], res['boxes'][:, 1], res['boxes'][:, 2], res['boxes'][:, 3],\
                                             res['boxes'][:, 4], res['boxes'][:, 5], res['boxes'][:, 6], res['boxes'][:, 7]

            x1, y1 = x1 * res['scales'][0], y1 * res['scales'][1]
            x2, y2 = x2 * res['scales'][0], y2 * res['scales'][1]
            x3, y3 = x3 * res['scales'][0], y3 * res['scales'][1]
            x4, y4 = x4 * res['scales'][0], y4 * res['scales'][1]

            boxes = np.transpose(np.stack([x1, y1, x2, y2, x3, y3, x4, y4]))

            if self.args.show_box:
                boxes = backward_convert(boxes, False)
                nake_name = res['image_id'].split('/')[-1]
                tools.makedirs(os.path.join(save_path, 'msra_td500_img_vis'))
                draw_path = os.path.join(save_path, 'msra_td500_img_vis', nake_name)
                draw_img = np.array(cv2.imread(res['image_id']), np.float32)

                drawer = DrawBox(self.cfgs)

                final_detections = drawer.draw_boxes_with_label_and_scores(draw_img,
                                                                           boxes=boxes,
                                                                           labels=res['labels'],
                                                                           scores=res['scores'],
                                                                           method=1,
                                                                           in_graph=False)
                cv2.imwrite(draw_path, final_detections)

            else:
                fw_txt_dt = open(os.path.join(save_path, 'msra_td500_res', 'res_{}.txt'.format(res['image_id'].split('/')[-1].split('.')[0]).replace('IMG', 'img')), 'w')

                for box in boxes:
                    line = '%d,%d,%d,%d,%d,%d,%d,%d\n' % (box[0], box[1], box[2], box[3],
                                                          box[4], box[5], box[6], box[7])
                    fw_txt_dt.write(line)
                fw_txt_dt.close()

                fw = open(txt_name, 'a+')
                fw.write('{}\n'.format(res['image_id'].split('/')[-1]))
                fw.close()

            pbar.set_description("Test image %s" % res['image_id'].split('/')[-1])

            pbar.update(1)

        for p in procs:
            p.join()

    def get_test_image(self):

        txt_name = '{}.txt'.format(self.cfgs.VERSION)
        if not self.args.show_box:
            if not os.path.exists(txt_name):
                fw = open(txt_name, 'w')
                fw.close()

            fr = open(txt_name, 'r')
            img_filter = fr.readlines()
            print('****************************' * 3)
            print('Already tested imgs:', img_filter)
            print('****************************' * 3)
            fr.close()

            test_imgname_list = [os.path.join(self.args.test_dir, img_name) for img_name in os.listdir(self.args.test_dir)
                                 if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff', '.JPG')) and
                                 (img_name + '\n' not in img_filter)]
        else:
            test_imgname_list = [os.path.join(self.args.test_dir, img_name) for img_name in os.listdir(self.args.test_dir)
                                 if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff', '.JPG'))]

        assert len(test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                            ' Note that, we only support img format of (.jpg, .png, and .tiff) '

        if self.args.num_imgs == np.inf:
            real_test_img_list = test_imgname_list
        else:
            real_test_img_list = test_imgname_list[: self.args.num_imgs]

        return real_test_img_list

