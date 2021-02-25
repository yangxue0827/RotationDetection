# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import


class PretrainModelZoo(object):
    def __init__(self):
        self.tf_zoo = ['resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'MobilenetV2', 'darknet',
                       'efficientnet-lite', 'efficientnet']
        self.pth_zoo = ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
                        'googlenet', 'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0',
                        'mnasnet1_3', 'mobilenet_v2', 'resnet101', 'resnet152', 'resnet18',
                        'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d',
                        'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5',
                        'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11',
                        'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19',
                        'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2']
        self.mxnet_zoo = ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d',
                          'resnet152_v1b', 'resnet101_v1b', 'resnet50_v1b', 'resnet34_v1b', 'resnet18_v1b']
        self.all_pretrain = {'tensorflow': self.tf_zoo, 'pytorch': self.pth_zoo, 'mxnet': self.mxnet_zoo}

    def pretrain_weight_path(self, net_name, root_path):
        if net_name in self.pth_zoo:
            weight_name = net_name
            weight_path = root_path + '/dataloader/pretrained_weights/' + weight_name + '.npy'
        elif net_name in self.tf_zoo or net_name in self.mxnet_zoo:
            if net_name.startswith("MobilenetV2"):
                weight_name = "mobilenet/mobilenet_v2_1.0_224"
            elif net_name.startswith("darknet"):
                weight_name = "darknet/darknet"
            elif net_name.startswith("efficientnet"):
                weight_name = "/efficientnet/{}/model".format(net_name)
            else:
                weight_name = net_name
            weight_path = root_path + '/dataloader/pretrained_weights/' + weight_name + '.ckpt'

        else:
            raise Exception('net name must in {}'.format(self.all_pretrain))

        return weight_path