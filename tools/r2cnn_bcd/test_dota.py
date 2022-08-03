# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append("../../")

from alpharotate.libs.models.detectors.r2cnn_bcd import build_whole_network
from tools.test_dota_base import TestDOTA
from configs import cfgs


class TestDOTAR2CNNBCD(TestDOTA):

    def eval(self):
        txt_name = '{}.txt'.format(self.cfgs.VERSION)
        real_test_img_list = self.get_test_image()

        r2cnn_bcd = build_whole_network.DetectionNetworkR2CNNBCD(cfgs=self.cfgs,
                                                                 is_training=False)
        self.test_dota(det_net=r2cnn_bcd, real_test_img_list=real_test_img_list, txt_name=txt_name)

        if not self.args.show_box:
            os.remove(txt_name)

if __name__ == '__main__':

    tester = TestDOTAR2CNNBCD(cfgs)
    tester.eval()


