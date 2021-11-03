# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append("../../")

from alpharotate.libs.models.detectors.atss import build_whole_network
from tools.test_dota_base import TestDOTA
from configs import cfgs


class TestDOTAATSS(TestDOTA):

    def eval(self):
        txt_name = '{}.txt'.format(self.cfgs.VERSION)
        real_test_img_list = self.get_test_image()

        atss = build_whole_network.DetectionNetworkATSS(cfgs=self.cfgs,
                                                        is_training=False)
        self.test_dota(det_net=atss, real_test_img_list=real_test_img_list, txt_name=txt_name)

        if not self.args.show_box:
            os.remove(txt_name)

if __name__ == '__main__':

    tester = TestDOTAATSS(cfgs)
    tester.eval()


