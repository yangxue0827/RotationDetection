import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "rotate_gpu_nms.hpp":
    void _rotate_nms(np.int32_t*, int*, np.float32_t*, int, int, float, int)

def rotate_gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float_t thresh, np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] \
        keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] \
        scores = dets[:, 5]
    cdef np.ndarray[np.int_t, ndim=1] \
        order = scores.argsort()[::-1]
    cdef np.ndarray[np.float32_t, ndim=2] \
        sorted_dets = dets[order, :]
    thresh = <np.float>thresh
    _rotate_nms(&keep[0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, thresh, device_id)
    keep = keep[:num_out]
    return order[keep]
