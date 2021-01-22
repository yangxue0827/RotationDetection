# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function


from libs.models.backbones import resnet, resnet_gluoncv, mobilenet_v2
from libs.models.backbones.efficientnet import efficientnet_builder, efficientnet_lite_builder
from libs.models.necks import fpn_p2top6, scrdet_neck


class BuildBackbone(object):

    def __init__(self, cfgs, is_training):
        self.cfgs = cfgs
        self.base_network_name = cfgs.NET_NAME
        self.is_training = is_training
        self.fpn_func = self.fpn_mode(cfgs.FPN_MODE)

    def fpn_mode(self, fpn_mode):
        """
        :param fpn_mode: 0-bifpn, 1-fpn
        :return:
        """

        if fpn_mode == 'fpn':
            fpn_func = fpn_p2top6.NeckFPN(self.cfgs)
        elif fpn_mode == 'scrdet':
            fpn_func = scrdet_neck.NeckSCRDet(self.cfgs)
        else:
            raise Exception('only support [fpn, scrdet]')
        return fpn_func

    def build_backbone(self, input_img_batch):

        if self.base_network_name.startswith('resnet_v1'):

            feature_dict = resnet.ResNetBackbone(self.cfgs).resnet_base(input_img_batch,
                                                                        scope_name=self.base_network_name,
                                                                        is_training=self.is_training)
            return self.fpn_func.fpn(feature_dict, self.is_training)

        elif self.base_network_name in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d',
                                        'resnet152_v1b', 'resnet101_v1b', 'resnet50_v1b', 'resnet34_v1b', 'resnet18_v1b']:

            feature_dict = resnet_gluoncv.ResNetGluonCVBackbone(self.cfgs).resnet_base(input_img_batch,
                                                                                       scope_name=self.base_network_name,
                                                                                       is_training=self.is_training)

            return self.fpn_func.fpn(feature_dict, self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):

            feature_dict = mobilenet_v2.MobileNetV2Backbone(self.cfgs).mobilenetv2_base(input_img_batch,
                                                                                        is_training=self.is_training)

            return self.fpn_func.fpn(feature_dict, self.is_training)

        elif 'efficientnet-lite' in self.base_network_name:
            feature_dict = efficientnet_lite_builder.EfficientNetLiteBackbone(self.cfgs).build_model_fpn_base(
                input_img_batch,
                model_name=self.base_network_name,
                training=True)

            return self.fpn_func.fpn(feature_dict, self.is_training)

        elif 'efficientnet' in self.base_network_name:
            feature_dict = efficientnet_builder.EfficientNetBackbone(self.cfgs).build_model_fpn_base(
                input_img_batch,
                model_name=self.base_network_name,
                training=True)
            return self.fpn_func.fpn(feature_dict, self.is_training)

        else:
            raise ValueError('Sorry, we only support resnet, mobilenet_v2 and efficient.')
