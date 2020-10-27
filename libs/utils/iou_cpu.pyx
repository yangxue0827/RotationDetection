# written by yjr

cimport cython
import numpy as np 
cimport numpy as np
import cv2
import time

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef bint BOOL

cdef DTYPE_t two_boxes_iou(np.ndarray[DTYPE_t, ndim=1] rectangle_1, np.ndarray[DTYPE_t, ndim=1] rectangle_2):

    """
	calu rectangle_1 and rectangle_2 iou
    :param rectangle_1: [x, y, w, h, theta]. shape: (5, )
    :param rectangle_2:
    :return:
    """
    cdef DTYPE_t area1 = rectangle_1[2] * rectangle_1[3]
    cdef DTYPE_t area2 = rectangle_2[2] * rectangle_2[3]

    rect_1 = ((rectangle_1[0], rectangle_1[1]), (rectangle_1[3], rectangle_1[2]), rectangle_1[-1])
    rect_2 = ((rectangle_2[0], rectangle_2[1]), (rectangle_2[3], rectangle_2[2]), rectangle_2[-1])

    inter_points = cv2.rotatedRectangleIntersection(rect_1, rect_2)[1]

    cdef np.ndarray[DTYPE_t, ndim=3] order_points
    cdef float inter_area, iou
    if inter_points is not None:
        order_points = cv2.convexHull(inter_points, returnPoints=True)

        inter_area = cv2.contourArea(order_points)
        if area1 + area2 == inter_area:
            print ("area1-->", area1)
            print ("area2-->", area2)
            print ("inter_area-->", inter_area)
        iou = inter_area *1.0 / (area1 + area2 - inter_area)
        return <DTYPE_t> iou
    else:
        return <DTYPE_t> 0.0

cpdef np.ndarray[DTYPE_t, ndim=2] get_iou_matrix(
    np.ndarray[DTYPE_t, ndim=2] boxes1, # (N, 5)
    np.ndarray[DTYPE_t, ndim=2] boxes2): # (M, 5)
    
    cdef unsigned int num_of_boxes1 = boxes1.shape[0]
    cdef unsigned int num_of_boxes2 = boxes2.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] iou_matrix = np.zeros((num_of_boxes1, num_of_boxes2), dtype=DTYPE)
    # cdef DTYPE_t box_iou 
    cdef unsigned int n, m
    # st = time.time()
    for n in range(num_of_boxes1):
        for m in range(num_of_boxes2):

            iou_matrix[n, m] = two_boxes_iou(boxes1[n], boxes2[m])
    # print "iou_matrix cost time: ", time.time() - st
    return iou_matrix

