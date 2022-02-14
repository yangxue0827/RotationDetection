# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append("../../")

from alpharotate.libs.models.detectors.kfiou import build_whole_network
from tools.test_msra_td500_base import TestIMSRATD500
from configs import cfgs


class TestMSRATD500KF(TestIMSRATD500):

    def eval(self):
        txt_name = '{}.txt'.format(self.cfgs.VERSION)
        real_test_img_list = self.get_test_image()

        kf = build_whole_network.DetectionNetworkKF(cfgs=self.cfgs,
                                                    is_training=False)
        self.test_msra_td500(det_net=kf, real_test_img_list=real_test_img_list, txt_name=txt_name)

        if not self.args.show_box:
            os.remove(txt_name)

if __name__ == '__main__':

    tester = TestMSRATD500KF(cfgs)
    tester.eval()


