# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("../../")

from alpharotate.libs.models.detectors.csl import build_whole_network
from tools.test_speed_base import Test
from configs import cfgs


class TestSpeed(Test):

    def eval(self):
        csl = build_whole_network.DetectionNetworkCSL(cfgs=self.cfgs,
                                                      is_training=False)

        all_boxes_r = self.eval_with_plac(img_dir=self.args.img_dir, det_net=csl,
                                          image_ext=self.args.image_ext)

        # imgs = os.listdir(self.args.img_dir)
        # real_test_imgname_list = [i.split(self.args.image_ext)[0] for i in imgs]

        # print(10 * "**")
        # print('rotation eval:')
        # evaler = EVAL(self.cfgs)
        # evaler.voc_evaluate_detections(all_boxes=all_boxes_r,
        #                                test_imgid_list=real_test_imgname_list,
        #                                test_annotation_path=self.args.test_annotation_path)


if __name__ == '__main__':

    tester = TestSpeed(cfgs)
    tester.eval()
