# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append("../../")

from alpharotate.libs.models.detectors.csl import build_whole_network
from tools.test_dota_base import TestDOTA
from configs import cfgs


class TestDOTACSL(TestDOTA):

    def eval(self):
        txt_name = '{}.txt'.format(self.cfgs.VERSION)
        real_test_img_list = self.get_test_image()

        csl = build_whole_network.DetectionNetworkCSL(cfgs=self.cfgs,
                                                      is_training=False)
        self.test_dota(det_net=csl, real_test_img_list=real_test_img_list, txt_name=txt_name)

        if not self.args.show_box:
            os.remove(txt_name)

if __name__ == '__main__':

    tester = TestDOTACSL(cfgs)
    tester.eval()


