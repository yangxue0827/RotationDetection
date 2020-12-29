# -*- coding:utf-8 -*-

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
sys.path.append("../../")

from libs.models.detectors.r3det import build_whole_network
from tools.test_hrsc2016_base import TestHRSC2016
from libs.configs import cfgs
from libs.val_libs.voc_eval_r import EVAL


class TestHRSC2016R3Det(TestHRSC2016):

    def eval(self):
        r3det = build_whole_network.DetectionNetworkR3Det(cfgs=self.cfgs,
                                                          is_training=False)

        all_boxes_r = self.eval_with_plac(img_dir=self.args.img_dir, det_net=r3det,
                                          image_ext=self.args.image_ext)

        # with open(cfgs.VERSION + '_detections_r.pkl', 'rb') as f2:
        #     all_boxes_r = pickle.load(f2)
        #
        #     print(len(all_boxes_r))

        imgs = os.listdir(self.args.img_dir)
        real_test_imgname_list = [i.split(self.args.image_ext)[0] for i in imgs]

        print(10 * "**")
        print('rotation eval:')
        evaler = EVAL(self.cfgs)
        evaler.voc_evaluate_detections(all_boxes=all_boxes_r,
                                       test_imgid_list=real_test_imgname_list,
                                       test_annotation_path=self.args.test_annotation_path)


if __name__ == '__main__':

    tester = TestHRSC2016R3Det(cfgs)
    tester.eval()
