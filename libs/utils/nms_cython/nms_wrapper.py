# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Linjie Deng
# --------------------------------------------------------
import torch

from libs.utils.nms_cython.cpu_nms import cpu_nms, cpu_soft_nms


def nms(dets, thresh, use_gpu=True):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    if dets.shape[1] == 5:
        raise NotImplementedError
    elif dets.shape[1] == 6:
        if torch.is_tensor(dets):
            dets = dets.cpu().detach().numpy()
        return cpu_nms(dets, thresh)
    else:
        raise NotImplementedError
