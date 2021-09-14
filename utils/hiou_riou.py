import cv2
import numpy as np


def hiou(boxes_1, boxes_2):

    """
    Calculate the Axis-aligned IoU

    :param boxes_1: horizontal bounding box1
    :param boxes_2: horizontal bounding box2
    :return: axis-aligned IoU
    """

    xmin_1, ymin_1, xmax_1, ymax_1 = np.split(boxes_1, 4, axis=1)

    xmin_2, ymin_2, xmax_2, ymax_2 = boxes_2[:, 0], boxes_2[:, 1], boxes_2[:, 2], boxes_2[:, 3]

    max_xmin = np.maximum(xmin_1, xmin_2)
    min_xmax = np.minimum(xmax_1, xmax_2)

    max_ymin = np.maximum(ymin_1, ymin_2)
    min_ymax = np.minimum(ymax_1, ymax_2)

    overlap_h = np.maximum(0., min_ymax - max_ymin)  # avoid h < 0
    overlap_w = np.maximum(0., min_xmax - max_xmin)

    overlaps = overlap_h * overlap_w

    area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
    area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

    iou = overlaps / (area_1 + area_2 - overlaps)

    return iou


def riou(boxes1, boxes2):
    """
    Calculate the Skew IoU

    :param boxes_1: rotated bounding box1
    :param boxes_2: rotated bounding box2
    :return: Skew IoU
    """

    ious = []
    if boxes1.shape[0] != 0:
        boxes1[:, 2] += 1.0
        boxes1[:, 3] += 1.0
        boxes2[:, 2] += 1.0
        boxes2[:, 3] += 1.0

        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]

        for i in range(boxes1.shape[0]):
            temp_ious = []
            r1 = ((boxes1[i][0], boxes1[i][1]), (boxes1[i][2], boxes1[i][3]), boxes1[i][4])
            r2 = ((boxes2[i][0], boxes2[i][1]), (boxes2[i][2], boxes2[i][3]), boxes2[i][4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[i] - int_area + 1e-4)

                # if boxes1[i][2] < 0.1 or boxes1[i][3] < 0.1 or boxes2[i][2] < 0.1 or boxes2[i][3] < 0.1:
                #     inter = 0

                inter = max(0.0, min(1.0, inter))

                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
            ious.append(temp_ious)

    return np.array(ious, dtype=np.float32)


if __name__ == '__main__':
    boxes1 = np.array([
        [50, 50, 70, 10, -45]], np.float32)

    boxes2 = np.array([
        [50, 25, 70, 10, -45]], np.float32)

    iou_r = riou(boxes1, boxes2)
    print(iou_r)

    from libs.utils.coordinate_convert import forward_convert
    boxes1 = forward_convert(boxes1, False)
    boxes2 = forward_convert(boxes2, False)

    x1_min = np.min(boxes1[:, ::2], axis=-1)
    y1_min = np.min(boxes1[:, 1::2], axis=-1)
    x1_max = np.max(boxes1[:, ::2], axis=-1)
    y1_max = np.max(boxes1[:, 1::2], axis=-1)
    boxes1 = np.transpose(np.stack([x1_min, y1_min, x1_max, y1_max]))

    x2_min = np.min(boxes2[:, ::2], axis=-1)
    y2_min = np.min(boxes2[:, 1::2], axis=-1)
    x2_max = np.max(boxes2[:, ::2], axis=-1)
    y2_max = np.max(boxes2[:, 1::2], axis=-1)
    boxes2 = np.transpose(np.stack([x2_min, y2_min, x2_max, y2_max]))

    iou_h = hiou(boxes1, boxes2)
    print(iou_h)

