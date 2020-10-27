import numpy as np
cimport numpy as np

cdef extern from "rbbox_overlaps.hpp":
    void _overlaps(np.float32_t*, np.float32_t*, np.float32_t*, int, int, int)

def rbbx_overlaps (np.ndarray[np.float32_t, ndim=2] boxes, np.ndarray[np.float32_t, ndim=2] query_boxes, np.int32_t device_id=0):
    # boxes: [x, y, w, h, theta]
    cdef int N = boxes.shape[0]
    cdef int K = query_boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] overlaps = np.zeros((N, K), dtype = np.float32)
    _overlaps(&overlaps[0, 0], &boxes[0, 0], &query_boxes[0, 0], N, K, device_id)
    return overlaps


