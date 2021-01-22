# -*-coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
# from tensorflow.python.tools import inspect_checkpoint as chkp
#
# # print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)


tf_mxnet_map = {}

tf_mxnet_prefix_map = {"weights": "weight",
                       "moving_mean": "running_mean",
                       "moving_variance": "running_var",
                       "beta": "beta",
                       "gamma": "gamma"}
def update_logitis():
    tf_mxnet_map["logits/weights"] = 'fc.weight'
    tf_mxnet_map["logits/biases"] = 'fc.bias'


def update_C1_resnet_v1_b():
    tmp_map = {"C1/conv0/weights": "conv1.weight",
               "C1/conv0/BatchNorm/beta": "bn1.beta",
               "C1/conv0/BatchNorm/gamma": "bn1.gamma",
               "C1/conv0/BatchNorm/moving_mean": "bn1.running_mean",
               "C1/conv0/BatchNorm/moving_variance": "bn1.running_var"}
    tf_mxnet_map.update(tmp_map)


def update_C1_resnet_v1_d():
    tmp_map = {"C1/conv0/weights": "conv1.0.weight",
               "C1/conv0/BatchNorm/beta": "conv1.1.beta",
               "C1/conv0/BatchNorm/gamma": "conv1.1.gamma",
               "C1/conv0/BatchNorm/moving_mean": "conv1.1.running_mean",
               "C1/conv0/BatchNorm/moving_variance": "conv1.1.running_var",
               "C1/conv1/weights": "conv1.3.weight",
               "C1/conv2/weights": "conv1.6.weight"}
    tf_mxnet_map.update(tmp_map)

    tf_prefix = "C1/conv1/BatchNorm/"
    for key in tf_mxnet_prefix_map.keys():
        if key != 'weights':
            tf_mxnet_map[tf_prefix+key] = "conv1.4." + tf_mxnet_prefix_map[key]

    tf_prefix = "C1/conv2/BatchNorm/"
    for key in tf_mxnet_prefix_map.keys():
        if key != 'weights':
            tf_mxnet_map[tf_prefix+key] = "bn1." + tf_mxnet_prefix_map[key]


def update_C2345(scope, bottleneck_nums):

    '''
    bottleneck nums :[3, 4, 6, 3] for res 50
    '''
    if '34' in scope or '18' in scope:
        conv_nums = 2
        downsample_flag = False
        cell_name = 'basicblock'
    else:
        conv_nums = 3
        downsample_flag = True
        cell_name = 'bottleneck'
    for layer, num in enumerate(bottleneck_nums):

        layer += 2  # 0->C2; 1->C3...3->C5
        for i in range(num):
            for j in range(conv_nums):
                tf_prefix = "C%d/%s_%d/conv%d/" % (layer, cell_name, i, j)
                for key in tf_mxnet_prefix_map.keys():
                    if key == 'weights':
                        tf_mxnet_map[tf_prefix + key] = "layer%d.%d.conv%d." % (layer-1, i, j + 1) + tf_mxnet_prefix_map[key]
                    else:
                        tf_mxnet_map[tf_prefix + "BatchNorm/" + key] = "layer%d.%d.bn%d." % (layer-1, i, j + 1) + tf_mxnet_prefix_map[key]
            if i == 0:
                tf_prefix = "C%d/%s_%d/shortcut/" % (layer, cell_name, i)
                for key in tf_mxnet_prefix_map.keys():
                    index = 1
                    if scope.endswith('b'):
                        index = 0
                    if downsample_flag or (layer-1)>1:
                        if key == 'weights':
                            tf_mxnet_map[tf_prefix + key] = "layer%d.%d.downsample.%d." % (layer-1, i, index) + tf_mxnet_prefix_map[key]
                        else:
                            tf_mxnet_map[tf_prefix + "BatchNorm/" + key] = "layer%d.%d.downsample.%d." % (layer-1, i, index+1) + tf_mxnet_prefix_map[key]



# def update_C2(bottleneck_num):
#     for i in range(bottleneck_num):
#         for j in range(3):
#             tf_prefix = "C2/bottleneck_%d/conv%d/" % (i, j)
#             for key in tf_mxnet_prefix_map.keys():
#                 if key == 'weights':
#                     tf_mxnet_map[tf_prefix+key] = "layer1.%d.conv%d." % (i, j+1) + tf_mxnet_prefix_map[key]
#                 else:
#                     tf_mxnet_map[tf_prefix + key] = "layer1.%d.bn%d." % (i, j + 1) + tf_mxnet_prefix_map[key]
#         if i==0:
#             tf_prefix = "C2/bottleneck_%d/shortcut/" % i
#             for key in tf_mxnet_prefix_map.keys():
#                 if key == 'weights':
#                     tf_mxnet_map[tf_prefix + key] = "layer1.%d.downsample.1." % i + tf_mxnet_prefix_map[key]
#                 else:
#                     tf_mxnet_map[tf_prefix + key] = "layer1.%d.downsample.2." % i + tf_mxnet_prefix_map[key]
#
# def update_C3(bottleneck_num):
#
#     for i in range(bottleneck_num):
#         for j in range(3):
#             tf_prefix = "C3/bottleneck_%d/conv%d/" % (i, j)
#             for key in tf_mxnet_prefix_map.keys():
#                 if key == 'weights':
#                     tf_mxnet_map[tf_prefix+key] = "layer2.%d.conv%d." % (i, j+1) + tf_mxnet_prefix_map[key]
#                 else:
#                     tf_mxnet_map[tf_prefix + key] = "layer2.%d.bn%d." % (i, j + 1) + tf_mxnet_prefix_map[key]
#         if i == 0:
#             tf_prefix = "C3/bottleneck_%d/shortcut/" % i
#             for key in tf_mxnet_prefix_map.keys():
#                 if key == 'weights':
#                     tf_mxnet_map[tf_prefix + key] = "layer2.%d.downsample.1." % i + tf_mxnet_prefix_map[key]
#                 else:
#                     tf_mxnet_map[tf_prefix + key] = "layer2.%d.downsample.2." % i + tf_mxnet_prefix_map[key]
#
# def update_C4(bottleneck_num):
#     for i in range(bottleneck_num):
#         for j in range(3):
#             tf_prefix = "C4/bottleneck_%d/conv%d/" % (i, j)
#             for key in tf_mxnet_prefix_map.keys():
#                 if key == 'weights':
#                     tf_mxnet_map[tf_prefix+key] = "layer3.%d.conv%d." % (i, j+1) + tf_mxnet_prefix_map[key]
#                 else:
#                     tf_mxnet_map[tf_prefix + key] = "layer3.%d.bn%d." % (i, j + 1) + tf_mxnet_prefix_map[key]
#         if i == 0:
#             tf_prefix = "C4/bottleneck_%d/shortcut/" % i
#             for key in tf_mxnet_prefix_map.keys():
#                 if key == 'weights':
#                     tf_mxnet_map[tf_prefix + key] = "layer3.%d.downsample.1." % i + tf_mxnet_prefix_map[key]
#                 else:
#                     tf_mxnet_map[tf_prefix + key] = "layer3.%d.downsample.2." % i + tf_mxnet_prefix_map[key]
#
# def update_C5(bottleneck_num):
#     for i in range(bottleneck_num):
#         for j in range(3):
#             tf_prefix = "C5/bottleneck_%d/conv%d/" % (i, j)
#             for key in tf_mxnet_prefix_map.keys():
#                 if key == 'weights':
#                     tf_mxnet_map[tf_prefix+key] = "layer4.%d.conv%d." % (i, j+1) + tf_mxnet_prefix_map[key]
#                 else:
#                     tf_mxnet_map[tf_prefix + key] = "layer4.%d.bn%d." % (i, j + 1) + tf_mxnet_prefix_map[key]
#         if i == 0:
#             tf_prefix = "C5/bottleneck_%d/shortcut/" % i
#             for key in tf_mxnet_prefix_map.keys():
#                 if key == 'weights':
#                     tf_mxnet_map[tf_prefix + key] = "layer4.%d.downsample.1." % i + tf_mxnet_prefix_map[key]
#                 else:
#                     tf_mxnet_map[tf_prefix + key] = "layer4.%d.downsample.2." % i + tf_mxnet_prefix_map[key]

def get_map(scope, bottleneck_nums, show_mxnettf=True, show_tfmxnet=True):

    if scope.endswith('b'):
        update_C1_resnet_v1_b()
    elif scope.endswith('d'):
        update_C1_resnet_v1_d()

    update_C2345(scope, bottleneck_nums)
    update_logitis()

    mxnet_tf_map = {}
    for tf_name, mxnet_name in tf_mxnet_map.items():
        mxnet_tf_map[mxnet_name] = tf_name

    if show_mxnettf:
        for key in sorted(mxnet_tf_map.keys()):
            print ("{} :: {}".format(key, mxnet_tf_map[key]))
        print(20*"===")

    if show_tfmxnet:
        for key in sorted(tf_mxnet_map.keys()):
            print("{} :: {}".format(key, tf_mxnet_map[key]))
        print(20 * "===")

    return tf_mxnet_map, mxnet_tf_map


if __name__ == "__main__":
    get_map(bottleneck_nums=[3, 4, 6, 3])
