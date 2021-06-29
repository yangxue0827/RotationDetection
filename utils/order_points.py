# some code from https://github.com/ming71/toolbox/blob/master/rotation/order_points.py

import os
import math
import cv2
import numpy as np


# this function is confined to rectangle
# clockwise, write by ming71
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


# this function can be used for polygon
def order_points_quadrangle(pts):
    from scipy.spatial import distance as dist
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left and bottom-left coordinate, use it as an
    # base vector to calculate the angles between the other two vectors

    vector_0 = np.array(bl - tl)
    vector_1 = np.array(rightMost[0] - tl)
    vector_2 = np.array(rightMost[1] - tl)

    angle = [np.arccos(cos_dist(vector_0, vector_1)), np.arccos(cos_dist(vector_0, vector_2))]
    (br, tr) = rightMost[np.argsort(angle), :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    # print(a, b)
    # print(zip(a, b))
    for a1, b1 in zip(a, b):
        part_up += a1 * b1
        a_sq += a1 ** 2
        b_sq += b1 ** 2
    part_down = math.sqrt(a_sq * b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down


############  another method##############

def sort_corners(quads):
    sorted_quads = np.zeros(quads.shape, dtype=np.float32)
    for i, corners in enumerate(quads):
        corners = corners.reshape(4, 2)
        centers = np.mean(corners, axis=0)
        corners = corners - centers
        cosine = corners[:, 0] / np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
        cosine = np.minimum(np.maximum(cosine, -1.0), 1.0)
        thetas = np.arccos(cosine) / np.pi * 180.0
        indice = np.where(corners[:, 1] > 0)[0]
        thetas[indice] = 360.0 - thetas[indice]
        corners = corners + centers
        corners = corners[thetas.argsort()[::-1], :]
        corners = corners.reshape(8)
        dx1, dy1 = (corners[4] - corners[0]), (corners[5] - corners[1])
        dx2, dy2 = (corners[6] - corners[2]), (corners[7] - corners[3])
        slope_1 = dy1 / dx1 if dx1 != 0 else np.iinfo(np.int32).max
        slope_2 = dy2 / dx2 if dx2 != 0 else np.iinfo(np.int32).max
        if slope_1 > slope_2:
            if corners[0] < corners[4]:
                first_idx = 0
            elif corners[0] == corners[4]:
                first_idx = 0 if corners[1] < corners[5] else 2
            else:
                first_idx = 2
        else:
            if corners[2] < corners[6]:
                first_idx = 1
            elif corners[2] == corners[6]:
                first_idx = 1 if corners[3] < corners[7] else 3
            else:
                first_idx = 3
        for j in range(4):
            idx = (first_idx + j) % 4
            sorted_quads[i, j * 2] = corners[idx * 2]
            sorted_quads[i, j * 2 + 1] = corners[idx * 2 + 1]
    return sorted_quads


# counterclockwise, write by WenQian
def re_order(bboxes, with_label=False):
    n=len(bboxes)
    targets=[]
    for i in range(n):
        box=bboxes[i]
        # 寻找x1
        x1=box[0]
        y1=box[1]
        x1_index=0
        for j in range(1,4):
            ### if x larger than x1 then continue
            if box[2*j]>x1:
                continue
            ### if x smaller than x1 then replace x1 as x
            elif box[2*j]<x1:
                x1=box[2*j]
                y1=box[2*j+1]
                x1_index=j
            ### if they are euqal then we aims to find the upper point
            else:
                if box[2*j+1]<y1:
                    x1=box[2*j]
                    y1=box[2*j+1]
                    x1_index=j
                else:
                    continue

        #寻找与x1连线中间点
        for j in range(4):
            if j==x1_index:
                continue
            x_=box[2*j]
            y_=box[2*j+1]
            x_index=j
            val=[]
            for k in range(4):
                if k==x_index or k==x1_index:
                    continue
                else:
                    x=box[2*k]
                    y=box[2*k+1]
                    if x1==x_:
                        val.append(x-x1)
                    else:
                        val1=(y-y1)-(y_-y1)/(x_-x1)*(x-x1)
                        val.append(val1)
            if val[0]*val[1]<0:
                x3=x_
                y3=y_
                for k in range(4):
                    if k==x_index or k==x1_index:
                        continue
                    x=box[2*k]
                    y=box[2*k+1]
                    if not x1==x3:
                        val=(y-y1)-(y3-y1)/(x3-x1)*(x-x1)
                        if val>=0:
                            x2=x
                            y2=y
                        if val<0:
                            x4=x
                            y4=y
                    else:
                        val=x1-x
                        if val>=0:
                            x2=x
                            y2=y
                        if val<0:
                            x4=x
                            y4=y
                break
        try:
            if with_label:
                targets.append([x1, y1, x2, y2, x3, y3, x4, y4, box[-1]])
            else:
                targets.append([x1, y1, x2, y2, x3, y3, x4, y4])
        except:
            print('**'*20)
            print(box)
            targets.append(box)
    return np.array(targets, np.float32)
# pts = np.array([[296, 245] ,[351 ,266], [208, 487],[263, 507]])
# npts = order_points_quadrangle(pts)

if __name__ == '__main__':
    pts = np.array([
        [242.7452, 314.5097, 242.7452, 133.4903, 333.2548, 133.4903, 333.2548, 314.5097],
        [333.2548, 133.4903, 333.2548, 314.5097, 242.7452, 314.5097, 242.7452, 133.4903],
        [60, 0, 80, 20, 20, 80, 0, 60],
        [40, 0, 40, 40, 0, 40, 0, 0]
    ])
    npts = sort_corners(pts)
    print(pts)
    # print(npts)
    print(re_order([[242.7452, 314.5097, 242.7452, 133.4903, 333.2548, 133.4903, 333.2548, 314.5097],
                    [333.2548, 133.4903, 333.2548, 314.5097, 242.7452, 314.5097, 242.7452, 133.4903],
                    [60, 0, 80, 20, 20, 80, 0, 60],
                    [40, 0, 40, 40, 0, 40, 0, 0]]))
