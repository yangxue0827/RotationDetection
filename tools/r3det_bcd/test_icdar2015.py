# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append("../../")

from alpharotate.libs.models.detectors.r3det_bcd import build_whole_network
from tools.test_icdar2015_base import TestICDAR2015
from configs import cfgs


class TestICDAR2015R3DetBCD(TestICDAR2015):

    def eval(self):
        txt_name = '{}.txt'.format(self.cfgs.VERSION)
        real_test_img_list = self.get_test_image()

        r3det_bcd = build_whole_network.DetectionNetworkR3DetBCD(cfgs=self.cfgs,
                                                                 is_training=False)
        self.test_icdar2015(det_net=r3det_bcd, real_test_img_list=real_test_img_list, txt_name=txt_name)

        if not self.args.show_box:
            os.remove(txt_name)

if __name__ == '__main__':

    tester = TestICDAR2015R3DetBCD(cfgs)
    tester.eval()


