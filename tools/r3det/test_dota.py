# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
sys.path.append("../../")

from libs.models.detectors.r3det import build_whole_network
from tools.test_dota_base import TestDOTA
from libs.configs import cfgs


class TestDOTAR3Det(TestDOTA):

    def eval(self):
        txt_name = '{}.txt'.format(self.cfgs.VERSION)
        real_test_img_list = self.get_test_image()

        r3det = build_whole_network.DetectionNetworkR3Det(cfgs=self.cfgs,
                                                          is_training=False)
        self.test_dota(det_net=r3det, real_test_img_list=real_test_img_list, txt_name=txt_name)

        if not self.args.show_box:
            os.remove(txt_name)

if __name__ == '__main__':

    tester = TestDOTAR3Det(cfgs)
    tester.eval()


