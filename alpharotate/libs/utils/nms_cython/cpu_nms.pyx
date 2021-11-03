# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------

import cv2
import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b

def cpu_soft_nms(
    np.ndarray[float, ndim=2] boxes,
    float thresh=0.3,
    unsigned int method=1,
    float sigma=0.5,
    float min_score=0.001
):
    cdef unsigned int N = boxes.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1, x2, y1, y2, t, s, w, h, xx, yy, tx1, tx2, ty1, ty2, tt, ts, tw, th, txx, tyy, area, weight, ov, inter

    inds = np.arange(N)
    for i in range(N):
        maxscore = boxes[i, 5]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        tt = boxes[i, 4]
        ts = boxes[i, 5]
        ti = inds[i]

        pos = i + 1
	    # get max box
        while pos < N:
            if maxscore < boxes[pos, 5]:
                maxscore = boxes[pos, 5]
                maxpos = pos
            pos = pos + 1

	    # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]
        boxes[i, 5] = boxes[maxpos, 5]
        inds[i] = inds[maxpos]

	    # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = tt
        boxes[maxpos, 5] = ts
        inds[maxpos] = ti

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        tt = boxes[i, 4]
        ts = boxes[i, 5]

        tw = tx2 - tx1
        th = ty2 - ty1
        txx = tx1 + tw * 0.5
        tyy = ty1 + th * 0.5

        pos = i + 1
	    # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            t = boxes[pos, 4]
            s = boxes[pos, 5]

            w = x2 - x1
            h = y2 - y1
            xx = x1 + w * 0.5
            yy = y1 + h * 0.5

            rtn, contours = cv2.rotatedRectangleIntersection(
                ((txx, tyy), (tw, th), tt),
                ((xx, yy), (w, h), t)
            )
            if rtn == 1:
                inter = np.round(np.abs(cv2.contourArea(contours)))
            elif rtn == 2:
                inter = min(tw * th, w * h)
            else:
                inter = 0.0

            if inter > 0.0:
                # iou between max box and detection box
                ov = inter / (tw * th + w * h - inter)
                if method == 1: # linear
                    if ov > thresh:
                        weight = 1 - ov
                    else:
                        weight = 1
                elif method == 2: # gaussian
                    weight = np.exp(-(ov * ov) / sigma)
                else: # original NMS
                    if ov > thresh:
                        weight = 0
                    else:
                        weight = 1
                boxes[pos, 5] = weight * boxes[pos, 5]
		        # if box score falls below threshold, discard the box by swapping with last box, update N
                if boxes[pos, 5] < min_score:
                    boxes[pos, 0] = boxes[N-1, 0]
                    boxes[pos, 1] = boxes[N-1, 1]
                    boxes[pos, 2] = boxes[N-1, 2]
                    boxes[pos, 3] = boxes[N-1, 3]
                    boxes[pos, 4] = boxes[N-1, 4]
                    boxes[pos, 5] = boxes[N-1, 5]
                    inds[pos] = inds[N - 1]
                    N = N - 1
                    pos = pos - 1
            pos = pos + 1

    return inds[:N]


def cpu_nms(
    np.ndarray[np.float32_t, ndim=2] dets,
    np.float thresh
):
    cdef np.ndarray[np.float32_t, ndim=1] ws = dets[:, 2] - dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] hs = dets[:, 3] - dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] xx = dets[:, 0] + ws * 0.5
    cdef np.ndarray[np.float32_t, ndim=1] yy = dets[:, 1] + hs * 0.5
    cdef np.ndarray[np.float32_t, ndim=1] tt = dets[:, 4]
    cdef np.ndarray[np.float32_t, ndim=1] areas = ws * hs

    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 5]
    cdef np.ndarray[np.intp_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = np.zeros((ndets), dtype=np.int)

    cdef int _i, _j, i, j, rtn
    cdef np.float32_t inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            rtn, contours = cv2.rotatedRectangleIntersection(
                ((xx[i], yy[i]), (ws[i], hs[i]), tt[i]),
                ((xx[j], yy[j]), (ws[j], hs[j]), tt[j])
            )
            if rtn == 1:
                inter = np.round(np.abs(cv2.contourArea(contours)))
            elif rtn == 2:
                inter = min(areas[i], areas[j])
            else:
                inter = 0.0
            ovr = inter / (areas[i] + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep
