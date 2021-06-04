# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import numpy as np

from PIL import Image, ImageDraw, ImageFont
import cv2
import random

from libs.label_name_dict.label_dict import LabelMap
from utils.tools import get_dota_short_names
from libs.utils.coordinate_convert import forward_convert
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo


class DrawBox(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs
        label_map = LabelMap(cfgs)
        self.name_label_map, self.label_name_map = label_map.name2label(), label_map.label2name()

        self.NOT_DRAW_BOXES = 0
        self.ONLY_DRAW_BOXES = -1
        self.ONLY_DRAW_BOXES_WITH_SCORES = -2

        self.STANDARD_COLORS = [
            'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
            'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
            'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
            'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
            'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
            'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
            'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
            'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
            'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
            'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
            'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
            'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
            'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
            'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
            'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
            'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
            'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
            'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
            'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
            'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
            'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
            'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
            'WhiteSmoke', 'Yellow', 'YellowGreen', 'LightBlue', 'LightGreen'
        ]
        self.FONT = ImageFont.load_default()

    def find_head_edge(self, box, head):
        head_dict = {0: '11', 1: '10', 2: '00', 3: '01'}
        flag = head_dict[int(head)]
        box[4] += random.random() * 0.1
        box_eight = forward_convert(np.array([box]), False)[0]
        box_eight = np.reshape(box_eight, [4, 2])
        four_edges = [[box_eight[0], box_eight[1]], [box_eight[1], box_eight[2]],
                      [box_eight[2], box_eight[3]], [box_eight[3], box_eight[0]]]
        for i in range(4):
            center_x = (four_edges[i][0][0] + four_edges[i][1][0]) / 2.
            center_y = (four_edges[i][0][1] + four_edges[i][1][1]) / 2.
            if (center_x - box[0]) >= 0 and (center_y - box[1]) >= 0:
                res = '11'
                if res == flag:
                    return four_edges[i]
            elif (center_x - box[0]) >= 0 and (center_y - box[1]) <= 0:
                res = '10'
                if res == flag:
                    return four_edges[i]
            elif (center_x - box[0]) <= 0 and (center_y - box[1]) <= 0:
                res = '00'
                if res == flag:
                    return four_edges[i]
            else:
                res = '01'
                if res == flag:
                    return four_edges[i]

    def draw_head(self, draw_obj, box, head, color, width=3):
        head_edge = self.find_head_edge(box, head)
        if head_edge is None:
            pass
        else:
            draw_obj.line(xy=[(head_edge[0][0], head_edge[0][1]), (head_edge[1][0], head_edge[1][1])],
                          fill='Red',
                          width=width+1)
            center_x = (head_edge[0][0] + head_edge[1][0]) / 2.
            center_y = (head_edge[0][1] + head_edge[1][1]) / 2.
            draw_obj.line(xy=[(center_x, center_y), (box[0], box[1])],
                          fill=color,
                          width=width)

    def draw_a_rectangel_in_img(self, draw_obj, box, color, width, method):
        '''
        use draw lines to draw rectangle. since the draw_rectangle func can not modify the width of rectangle
        :param draw_obj:
        :param box: [x1, y1, x2, y2]
        :return:
        '''
        # color = (0, 255, 0)
        if method == 0:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            top_left, top_right = (x1, y1), (x2, y1)
            bottom_left, bottom_right = (x1, y2), (x2, y2)

            draw_obj.line(xy=[top_left, top_right],
                          fill=color,
                          width=width)
            draw_obj.line(xy=[top_left, bottom_left],
                          fill=color,
                          width=width)
            draw_obj.line(xy=[bottom_left, bottom_right],
                          fill=color,
                          width=width)
            draw_obj.line(xy=[top_right, bottom_right],
                          fill=color,
                          width=width)
        elif method == 1:
            x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            draw_obj.line(xy=[(rect[0][0], rect[0][1]), (rect[1][0], rect[1][1])],
                          fill=color,
                          width=width)
            draw_obj.line(xy=[(rect[1][0], rect[1][1]), (rect[2][0], rect[2][1])],
                          fill=color,
                          width=width)
            draw_obj.line(xy=[(rect[2][0], rect[2][1]), (rect[3][0], rect[3][1])],
                          fill=color,
                          width=width)
            draw_obj.line(xy=[(rect[3][0], rect[3][1]), (rect[0][0], rect[0][1])],
                          fill=color,
                          width=width)
        elif method == 2:
            for ii in range(box.shape[0] // 2):
                draw_obj.line(xy=[(box[(2 * ii) % box.shape[0]], box[(2 * ii + 1) % box.shape[0]]),
                                  (box[(2 * ii + 2) % box.shape[0]], box[(2 * ii + 3) % box.shape[0]])],
                              fill=color,
                              width=width)
        else:
            pass

    def only_draw_scores(self, draw_obj, box, score, color):

        x, y = box[0], box[1]
        draw_obj.rectangle(xy=[x, y, x+60, y+10],
                           fill=color)
        draw_obj.text(xy=(x, y),
                      text="obj:" + str(round(score, 2)),
                      fill='black',
                      font=self.FONT)

    def draw_label_with_scores(self, draw_obj, box, label, score, color):
        x, y = box[0], box[1]
        draw_obj.rectangle(xy=[x, y, x + 60, y + 10],
                           fill=color)

        txt = self.label_name_map[label] + ':' + str(round(score, 2))
        draw_obj.text(xy=(x, y),
                      text=txt,
                      fill='black',
                      font=self.FONT)

    def draw_label_with_scores_csl(self, draw_obj, box, label, score, method, head, color='White'):
        x, y = box[0], box[1]
        draw_obj.rectangle(xy=[x, y, x + 60, y + 10],
                           fill='White')

        if self.cfgs.DATASET_NAME.startswith('DOTA'):
            label_name = get_dota_short_names(self.label_name_map[label])
        else:
            label_name = self.label_name_map[label]
        txt = label_name + ':' + str(round(score, 2))
        # txt = ' ' + label_name
        draw_obj.text(xy=(x, y),
                      text=txt,
                      fill='black',
                      font=self.FONT)
        if method == 1:
            draw_obj.rectangle(xy=[x, y + 10, x + 60, y + 20],
                               fill='White')
            if self.cfgs.ANGLE_RANGE == 180:
                if box[2] < box[3]:
                    angle = box[-1] + 90
                else:
                    angle = box[-1]
            else:
                angle = box[-1]
            txt_angle = 'angle:%.1f' % angle
            # txt_angle = ' %.1f' % angle
            draw_obj.text(xy=(x, y + 10),
                          text=txt_angle,
                          fill='black',
                          font=self.FONT)
            if head != -1:
                draw_obj.rectangle(xy=[x, y + 20, x + 60, y + 30],
                                   fill='White')
                txt_head = 'head:%d' % head
                draw_obj.text(xy=(x, y + 20),
                              text=txt_head,
                              fill='black',
                              font=self.FONT)
                self.draw_head(draw_obj, box, head, color)

    def draw_boxes_with_label_and_scores(self, img_array, boxes, labels, scores, method, head=None, is_csl=False, in_graph=True):
        if in_graph:
            pretrain_zoo = PretrainModelZoo()
            if self.cfgs.NET_NAME in pretrain_zoo.pth_zoo or self.cfgs.NET_NAME in pretrain_zoo.mxnet_zoo:
                img_array = (img_array * np.array(self.cfgs.PIXEL_STD) + np.array(self.cfgs.PIXEL_MEAN_)) * 255
            else:
                img_array = img_array + np.array(self.cfgs.PIXEL_MEAN)
        if method == 3:
            img_array = self.draw_contours(img_array, boxes, labels)
        elif method == 4:
            img_array = self.draw_boxes_ellipse(img_array, boxes, labels)

        img_array.astype(np.float32)
        boxes = boxes.astype(np.float32)
        labels = labels.astype(np.int32)
        img_array = np.array(img_array * 255 / np.max(img_array), dtype=np.uint8)

        img_obj = Image.fromarray(img_array)
        raw_img_obj = img_obj.copy()

        draw_obj = ImageDraw.Draw(img_obj)
        num_of_objs = 0

        if head is None:
            head = np.ones_like(labels) * -1

        for box, a_label, a_score, a_head in zip(boxes, labels, scores, head):

            if a_label != self.NOT_DRAW_BOXES:
                num_of_objs += 1
                self.draw_a_rectangel_in_img(draw_obj, box, color=self.STANDARD_COLORS[a_label], width=3, method=method)
                if a_label == self.ONLY_DRAW_BOXES:  # -1
                    continue
                elif a_label == self.ONLY_DRAW_BOXES_WITH_SCORES:  # -2
                     self.only_draw_scores(draw_obj, box, a_score, color='White')
                else:
                    if is_csl:
                        self.draw_label_with_scores_csl(draw_obj, box, a_label, a_score, method, a_head, color='White')
                    else:
                        self.draw_label_with_scores(draw_obj, box, a_label, a_score, color='White')

        out_img_obj = Image.blend(raw_img_obj, img_obj, alpha=0.7)

        return np.array(out_img_obj)

    def draw_boxes_ellipse(self, img_array, boxes, labels):
        labels = labels.astype(np.int32)
        for box, a_label in zip(boxes, labels):
            if box[2] < box[3]:
                box[4] += 90
            img_array = cv2.ellipse(img_array, center=(box[0], box[1]),
                                    axes=(max(box[2], box[3]), min(box[2], box[3])),
                                    angle=box[4], startAngle=0, endAngle=360,
                                    color=(255, 0, 0),
                                    thickness=3, lineType=-1)
        return img_array

    def draw_contours(self, img_array, polys, labels):
        for poly, a_label in zip(polys, labels):
            img_array = cv2.fillConvexPoly(img_array, np.reshape(poly, [-1, 2]).astype(np.int32), color=(255, 255, 255))
        return img_array








