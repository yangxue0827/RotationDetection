from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from shapely.geometry import Polygon

from libs.utils.coordinate_convert import backward_convert
from libs.models.samplers.samper import Sampler


class SamplerFCOS(Sampler):

    def __init__(self, cfgs):
        super(SamplerFCOS, self).__init__(cfgs)
        # self.min_obj_size = 10

    def fit_line(self, p1, p2):
        # fit a line ax+by+c = 0
        if p1[0] == p1[1]:
            return [1., 0., -p1[0]]
        else:
            [k, b] = np.polyfit(p1, p2, deg=1)
            return [k, -1., b]

    def point_dist_to_line_(self, p1, p2, p3):
        # compute the distance from p3 to p1-p2
        return np.linalg.norm(np.reshape(np.cross(p2 - p1, p1 - p3), [-1, 1]), axis=-1) / np.linalg.norm(p2 - p1, axis=-1)

    def point_dist_to_line(self, p1, p2, p3):
        # compute the distance from p3 to p1-p2
        return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

    def shrink_poly(self, poly, r):
        '''
        fit a poly inside the origin poly, maybe bugs here...
        used for generate the score map
        :param poly: the text poly
        :param r: r in the paper
        :return: the shrinked poly
        '''
        # shrink ratio
        R = 0.3
        # find the longer pair
        if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                        np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
            # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
            ## p0, p3
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
        else:
            ## p0, p3
            # print poly
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
        return poly

    def line_cross_point(self, line1, line2):
        # line1 0= ax+by+c, compute the cross point of line1 and line2
        if line1[0] != 0 and line1[0] == line2[0]:
            print('Cross point does not exist')
            return None
        if line1[0] == 0 and line2[0] == 0:
            print('Cross point does not exist')
            return None
        if line1[1] == 0:
            x = -line1[2]
            y = line2[0] * x + line2[2]
        elif line2[1] == 0:
            x = -line2[2]
            y = line1[0] * x + line1[2]
        else:
            k1, _, b1 = line1
            k2, _, b2 = line2
            x = -(b1 - b2) / (k1 - k2)
            y = k1 * x + b1
        return np.array([x, y], dtype=np.float32)

    def line_verticle(self, line, point):
        # get the verticle line from line across point
        if line[1] == 0:
            verticle = [0, -1, point[1]]
        else:
            if line[0] == 0:
                verticle = [1, 0, -point[0]]
            else:
                verticle = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
        return verticle

    def rectangle_from_parallelogram(self, poly):
        '''
        fit a rectangle from a parallelogram
        :param poly:
        :return:
        '''
        p0, p1, p2, p3 = poly
        angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
        if angle_p0 < 0.5 * np.pi:
            if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
                # p0 and p2
                ## p0
                p2p3 = self.fit_line([p2[0], p3[0]], [p2[1], p3[1]])
                p2p3_verticle = self.line_verticle(p2p3, p0)

                new_p3 = self.line_cross_point(p2p3, p2p3_verticle)
                ## p2
                p0p1 = self.fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                p0p1_verticle = self.line_verticle(p0p1, p2)

                new_p1 = self.line_cross_point(p0p1, p0p1_verticle)
                return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
            else:
                p1p2 = self.fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                p1p2_verticle = self.line_verticle(p1p2, p0)

                new_p1 = self.line_cross_point(p1p2, p1p2_verticle)
                p0p3 = self.fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                p0p3_verticle = self.line_verticle(p0p3, p2)

                new_p3 = self.line_cross_point(p0p3, p0p3_verticle)
                return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
                # p1 and p3
                ## p1
                p2p3 = self.fit_line([p2[0], p3[0]], [p2[1], p3[1]])
                p2p3_verticle = self.line_verticle(p2p3, p1)

                new_p2 = self.line_cross_point(p2p3, p2p3_verticle)
                ## p3
                p0p1 = self.fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                p0p1_verticle = self.line_verticle(p0p1, p3)

                new_p0 = self.line_cross_point(p0p1, p0p1_verticle)
                return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
            else:
                p0p3 = self.fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                p0p3_verticle = self.line_verticle(p0p3, p1)

                new_p0 = self.line_cross_point(p0p3, p0p3_verticle)
                p1p2 = self.fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                p1p2_verticle = self.line_verticle(p1p2, p3)

                new_p2 = self.line_cross_point(p1p2, p1p2_verticle)
                return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)

    def sort_rectangle(self, poly):
        # sort the four coordinates of the polygon, points in poly should be sorted clockwise
        # First find the lowest point
        p_lowest = np.argmax(poly[:, 1])
        if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
            # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
            p0_index = np.argmin(np.sum(poly, axis=1))
            p1_index = (p0_index + 1) % 4
            p2_index = (p0_index + 2) % 4
            p3_index = (p0_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
        else:
            # 找到最低点右边的点 - find the point that sits right to the lowest point
            p_lowest_right = (p_lowest - 1) % 4
            p_lowest_left = (p_lowest + 1) % 4
            angle = np.arctan(
                -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
            # assert angle > 0
            # if angle <= 0:
                # print(angle, poly[p_lowest], poly[p_lowest_right])
            if angle / np.pi * 180 > 45:
                # 这个点为p2 - this point is p2
                p2_index = p_lowest
                p1_index = (p2_index - 1) % 4
                p0_index = (p2_index - 2) % 4
                p3_index = (p2_index + 1) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
            else:
                # 这个点为p3 - this point is p3
                p3_index = p_lowest
                p0_index = (p3_index + 1) % 4
                p1_index = (p3_index + 2) % 4
                p2_index = (p3_index + 3) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], angle

    def generate_rbox(self, im_size, poly, label):
        h, w = im_size
        label_map = np.zeros((h, w), dtype=np.uint8)
        score_map = np.zeros((h, w), dtype=np.uint8)
        geo_map = np.zeros((h, w, 5), dtype=np.float32)
        # mask used during traning, to ignore some hard areas
        # training_mask = np.ones((h, w), dtype=np.uint8)
        if label != 0:

            r = [None, None, None, None]
            for i in range(4):
                r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                           np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
            # score map
            # shrinked_poly = self.shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
            shrinked_poly = np.array([poly])
            cv2.fillPoly(label_map, shrinked_poly, int(label))
            cv2.fillPoly(score_map, shrinked_poly, 1)
            # if the poly is too small, then ignore it during training
            # poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
            # poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
            # if min(poly_h, poly_w) < self.min_obj_size:
            #     cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

            xy_in_poly = np.argwhere(label_map == label)
            # if geometry == 'RBOX':
            # 对任意两个顶点的组合生成一个平行四边形 - generate a parallelogram for any combination of two vertices
            fitted_parallelograms = []
            for i in range(4):
                p0 = poly[i]
                p1 = poly[(i + 1) % 4]
                p2 = poly[(i + 2) % 4]
                p3 = poly[(i + 3) % 4]
                edge = self.fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                backward_edge = self.fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                forward_edge = self.fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                if self.point_dist_to_line(p0, p1, p2) > self.point_dist_to_line(p0, p1, p3):
                    # 平行线经过p2 - parallel lines through p2
                    if edge[1] == 0:
                        edge_opposite = [1, 0, -p2[0]]
                    else:
                        edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
                else:
                    # 经过p3 - after p3
                    if edge[1] == 0:
                        edge_opposite = [1, 0, -p3[0]]
                    else:
                        edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
                # move forward edge
                new_p0 = p0
                new_p1 = p1
                new_p2 = p2
                new_p3 = p3
                new_p2 = self.line_cross_point(forward_edge, edge_opposite)
                if self.point_dist_to_line(p1, new_p2, p0) > self.point_dist_to_line(p1, new_p2, p3):
                    # across p0
                    if forward_edge[1] == 0:
                        forward_opposite = [1, 0, -p0[0]]
                    else:
                        forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
                else:
                    # across p3
                    if forward_edge[1] == 0:
                        forward_opposite = [1, 0, -p3[0]]
                    else:
                        forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
                new_p0 = self.line_cross_point(forward_opposite, edge)
                new_p3 = self.line_cross_point(forward_opposite, edge_opposite)
                fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
                # or move backward edge
                new_p0 = p0
                new_p1 = p1
                new_p2 = p2
                new_p3 = p3
                new_p3 = self.line_cross_point(backward_edge, edge_opposite)
                if self.point_dist_to_line(p0, p3, p1) > self.point_dist_to_line(p0, p3, p2):
                    # across p1
                    if backward_edge[1] == 0:
                        backward_opposite = [1, 0, -p1[0]]
                    else:
                        backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
                else:
                    # across p2
                    if backward_edge[1] == 0:
                        backward_opposite = [1, 0, -p2[0]]
                    else:
                        backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
                new_p1 = self.line_cross_point(backward_opposite, edge)
                new_p2 = self.line_cross_point(backward_opposite, edge_opposite)
                fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            areas = [Polygon(t).area for t in fitted_parallelograms]
            parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
            # sort the polygon
            parallelogram_coord_sum = np.sum(parallelogram, axis=1)
            min_coord_idx = np.argmin(parallelogram_coord_sum)
            parallelogram = parallelogram[
                [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

            rectange = self.rectangle_from_parallelogram(parallelogram)
            rectange, rotate_angle = self.sort_rectangle(rectange)

            p0_rect, p1_rect, p2_rect, p3_rect = rectange
            # tmp = geo_map[xy_in_poly[:, 0], xy_in_poly[:, 1], 0]
            # tmp1 = self.point_dist_to_line_(np.tile(p0_rect[np.newaxis, :], [tmp.shape[0], 1]), np.tile(p1_rect[np.newaxis, :], [tmp.shape[0], 1]), xy_in_poly[:, ::-1])
            # for y, x in xy_in_poly:
            #     point = np.array([x, y], dtype=np.float32)
            #     # top
            #     geo_map[y, x, 0] = self.point_dist_to_line(p0_rect, p1_rect, point)
            #     # right
            #     geo_map[y, x, 1] = self.point_dist_to_line(p1_rect, p2_rect, point)
            #     # down
            #     geo_map[y, x, 2] = self.point_dist_to_line(p2_rect, p3_rect, point)
            #     # left
            #     geo_map[y, x, 3] = self.point_dist_to_line(p3_rect, p0_rect, point)
            #     # angle
            #     geo_map[y, x, 4] = rotate_angle
            # top
            geo_map[xy_in_poly[:, 0], xy_in_poly[:, 1], 0] = self.point_dist_to_line_(p0_rect, p1_rect, xy_in_poly[:, ::-1])
            # right
            geo_map[xy_in_poly[:, 0], xy_in_poly[:, 1], 1] = self.point_dist_to_line_(p1_rect, p2_rect, xy_in_poly[:, ::-1])
            # down
            geo_map[xy_in_poly[:, 0], xy_in_poly[:, 1], 2] = self.point_dist_to_line_(p2_rect, p3_rect, xy_in_poly[:, ::-1])
            # left
            geo_map[xy_in_poly[:, 0], xy_in_poly[:, 1], 3] = self.point_dist_to_line_(p3_rect, p0_rect, xy_in_poly[:, ::-1])
            # angle
            geo_map[xy_in_poly[:, 0], xy_in_poly[:, 1], 4] = rotate_angle

        return geo_map, score_map

    def get_rectangle(self, points, geometry):
        pointx, pointy = points[:, 0], points[:, 1]
        top, right, down, left, theta = geometry[:, 0], geometry[:, 1], geometry[:, 2], geometry[:, 3], geometry[:, 4]
        xlt, ylt = pointx - left, pointy - top
        xld, yld = pointx - left, pointy + down
        xrd, yrd = pointx + right, pointy + down
        xrt, yrt = pointx + right, pointy - top

        xlt_ = np.cos(theta) * (xlt - pointx) + np.sin(theta) * (ylt - pointy) + pointx
        ylt_ = -np.sin(theta) * (xlt - pointx) + np.cos(theta) * (ylt - pointy) + pointy

        xrt_ = np.cos(theta) * (xrt - pointx) + np.sin(theta) * (yrt - pointy) + pointx
        yrt_ = -np.sin(theta) * (xrt - pointx) + np.cos(theta) * (yrt - pointy) + pointy

        xld_ = np.cos(theta) * (xld - pointx) + np.sin(theta) * (yld - pointy) + pointx
        yld_ = -np.sin(theta) * (xld - pointx) + np.cos(theta) * (yld - pointy) + pointy

        xrd_ = np.cos(theta) * (xrd - pointx) + np.sin(theta) * (yrd - pointy) + pointx
        yrd_ = -np.sin(theta) * (xrd - pointx) + np.cos(theta) * (yrd - pointy) + pointy

        convert_box = np.transpose(np.stack([xlt_, ylt_, xrt_, yrt_, xrd_, yrd_, xld_, yld_], axis=0))
        return convert_box

    def fcos_target(self, gt_boxes_labels, image, fm_size_list):

        """
        :param gt_boxes_labels: [-1, 9]
        :param image_batch: [h, w, 3]
        :param fm_size_list: [-1, 2]
        :return: [-1, 4 + 1 + 1]
        """

        gt_boxes_labels = np.array(gt_boxes_labels, np.int32)
        # raw_height, raw_width = image_batch.shape[:2]

        gt_boxes_labels = np.concatenate([np.zeros((1, 9)), gt_boxes_labels])
        gtboxes_5 = backward_convert(gt_boxes_labels)
        gtboxes_area = gtboxes_5[:, 2] * gtboxes_5[:, 3]
        gt_boxes_labels = gt_boxes_labels[np.argsort(gtboxes_area)]
        boxes_cnt = len(gt_boxes_labels)

        offset_list = []
        for i in range(boxes_cnt):
            offset_i, score_i = self.generate_rbox(image.shape[:-1],
                                                   np.array(np.reshape(gt_boxes_labels[i, :-1], [4, 2]), np.int32),
                                                   gt_boxes_labels[i, -1])
            offset_list.append(offset_i[:, :, np.newaxis, :])
        offset = np.concatenate(offset_list, axis=2)

        center = (
        (np.minimum(offset[:, :, :, 3], offset[:, :, :, 1]) * np.minimum(offset[:, :, :, 0], offset[:, :, :, 2])) / (
            np.maximum(offset[:, :, :, 3], offset[:, :, :, 1]) * np.maximum(offset[:, :, :, 0], offset[:, :, :, 2]) + self.cfgs.EPSILON))
        center = np.sqrt(np.abs(center))
        center[:, :, 0] = 0

        cls = gt_boxes_labels[:, 8]

        cls_res_list = []
        ctr_res_list = []
        geo_res_list = []

        for fm_i, stride in enumerate(self.cfgs.ANCHOR_STRIDE):
            fm_height = fm_size_list[fm_i][0]
            fm_width = fm_size_list[fm_i][1]

            shift_x = np.arange(0, fm_width)
            shift_y = np.arange(0, fm_height)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            xy = np.vstack((shift_y.ravel(), shift_x.ravel())).transpose()

            off_xy = offset[xy[:, 0] * stride, xy[:, 1] * stride]
            # off_xy = offset[xy[:, 0] * stride + int(0.5 * stride), xy[:, 1] * stride + int(0.5 * stride)]

            off_max_xy = off_xy[:, :, :-1].max(axis=2)
            off_valid = np.zeros((fm_height, fm_width, boxes_cnt))

            is_in_boxes = (off_xy[:, :, :-1] > 0).all(axis=2)
            is_in_layer = (off_max_xy <= self.cfgs.SET_WIN[fm_i + 1]) & \
                          (off_max_xy >= self.cfgs.SET_WIN[fm_i])
            off_valid[xy[:, 0], xy[:, 1], :] = is_in_boxes & is_in_layer
            off_valid[:, :, 0] = 0

            hit_gt_ind = np.argmax(off_valid, axis=2)

            # geo
            geo_res = np.zeros((fm_height, fm_width, 5))
            geo_res[xy[:, 0], xy[:, 1]] = offset[xy[:, 0] * stride, xy[:, 1] * stride, hit_gt_ind[xy[:, 0], xy[:, 1]]]
            geo_res_list.append(geo_res.reshape(-1, 5))

            # cls
            cls_res = np.zeros((fm_height, fm_width))
            cls_res[xy[:, 0], xy[:, 1]] = cls[hit_gt_ind[xy[:, 0], xy[:, 1]]]
            # cv2.imwrite('./cls_{}.jpg'.format(stride), cls_res * 255)
            cls_res_list.append(cls_res.reshape(-1))

            # centerness
            center_res = np.zeros((fm_height, fm_width))
            center_res[xy[:, 0], xy[:, 1]] = center[
                xy[:, 0] * stride, xy[:, 1] * stride,
                hit_gt_ind[xy[:, 0], xy[:, 1]]]
            center_res[xy[:, 0], xy[:, 1]] = center[
                xy[:, 0] * stride + int(0.5 * stride), xy[:, 1] * stride + int(0.5 * stride),
                hit_gt_ind[xy[:, 0], xy[:, 1]]]
            # cv2.imwrite('./centerness_{}.jpg'.format(stride), center_res * 255)
            ctr_res_list.append(center_res.reshape(-1))

        cls_res_final = np.concatenate(cls_res_list, axis=0)[:, np.newaxis]
        ctr_res_final = np.concatenate(ctr_res_list, axis=0)[:, np.newaxis]
        geo_res_final = np.concatenate(geo_res_list, axis=0)
        return np.concatenate(
            [cls_res_final, ctr_res_final, geo_res_final], axis=1)

    def get_fcos_target_batch(self, gtboxes_batch, img_batch, fm_size_list):
        fcos_target_batch = []
        for i in range(self.cfgs.BATCH_SIZE):
            gt_target = self.fcos_target(gtboxes_batch[i, :, :], img_batch[i, :, :, :], fm_size_list)
            fcos_target_batch.append(gt_target)
        return np.array(fcos_target_batch, np.float32)


if __name__ == '__main__':
    from libs.configs import cfgs

    sampler = SamplerFCOS(cfgs)

    # geo_map, score_map = sampler.generate_rbox([80, 80], np.array([[10, 0], [30, 20], [20, 30], [0, 10]]), 1)
    # geo_map, score_map = sampler.generate_rbox([80, 80], np.array([[20, 0], [30, 10], [10, 30], [0, 20]]), 1)
    print(np.array([[20, 0], [40, 20], [20, 40], [0, 20]])[::-1, :])
    geo_map, score_map = sampler.generate_rbox([80, 80], np.array([[20, 0], [40, 20], [20, 40], [0, 20]])[::-1, :], 1)

    print(score_map.shape)
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    yx_text = np.argwhere(score_map > 0.8)
    # sort the text boxes via the y axis
    yx_text = yx_text[np.argsort(yx_text[:, 0])]
    # restore
    boxes = sampler.get_rectangle(yx_text[:, ::-1], geo_map[yx_text[:, 0], yx_text[:, 1], :])
    print(boxes[1:10])
    # image = np.zeros([512, 512, 3])
    # sampler.fcos_target(np.array([[60, 0, 80, 20, 20, 80, 0, 60, 1],
    #                               [160, 100, 180, 120, 120, 180, 100, 160, 4],
    #                               [15, 0, 20, 5, 5, 20, 0, 15, 2],
    #                               [120, 0, 160, 40, 40, 160, 0, 120, 5],
    #                               [240, 0, 320, 80, 80, 320, 0, 240, 1],
    #                               [360, 0, 480, 160, 160, 480, 0, 360, 1]]),
    #                     image,
    #                     [[64, 64], [32, 32], [16, 16], [8, 8], [4, 4]])
