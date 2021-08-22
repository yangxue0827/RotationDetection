# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import


class_names = [
        'back_ground', 'person', 'bicycle', 'car', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

classes_originID = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
    'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9,
    'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
    'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17,
    'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22,
    'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
    'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33,
    'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37,
    'kite': 38, 'baseball bat': 39, 'baseball glove': 40,
    'skateboard': 41, 'surfboard': 42, 'tennis racket': 43,
    'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48,
    'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53,
    'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57,
    'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61,
    'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65,
    'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73,
    'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77,
    'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81,
    'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86,
    'scissors': 87, 'teddy bear': 88, 'hair drier': 89,
    'toothbrush': 90}


class LabelMap(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs

    def coco_name2abel(self):
        # originID_classes = {item: key for key, item in classes_originID.items()}
        name_label_map = dict(zip(class_names, range(len(class_names))))
        return name_label_map

    def name2label(self):

        if self.cfgs.DATASET_NAME == 'WIDER':
            name_label_map = {
                'back_ground': 0,
                'face': 1
            }
        elif self.cfgs.DATASET_NAME in ['ICDAR2015', 'MSRA-TD500', 'MLT', 'Total_Text']:
            name_label_map = {
                'back_ground': 0,
                'text': 1
            }
        elif self.cfgs.DATASET_NAME == 'HRSC2016':
            name_label_map = {
                'back_ground': 0,
                'ship': 1
            }
        elif self.cfgs.DATASET_NAME.startswith('OHD-SJTU-ALL'):
            name_label_map = {
                'back_ground': 0,
                'small-vehicle': 1,
                'ship': 2,
                'plane': 3,
                'large-vehicle': 4,
                'helicopter': 5,
                'harbor': 6,
            }
        elif self.cfgs.DATASET_NAME.startswith('OHD-SJTU'):
            name_label_map = {
                'back_ground': 0,
                'ship': 1,
                'plane': 2
            }
        elif self.cfgs.DATASET_NAME.startswith('SSDD++'):
            name_label_map = {
                'back_ground': 0,
                'ship': 1
            }
        elif self.cfgs.DATASET_NAME.startswith('SKU110K-R'):
            name_label_map = {
                'back_ground': 0,
                'commodity': 1
            }
        elif self.cfgs.DATASET_NAME.startswith('UCAS-AOD'):
            name_label_map = {
                'back_ground': 0,
                'car': 1,
                'plane': 2
            }
        elif self.cfgs.DATASET_NAME.startswith('DOTA'):
            name_label_map = {
                'car': 0
            }
            if self.cfgs.DATASET_NAME == 'DOTA1.5':
                name_label_map['container-crane'] = 16
            if self.cfgs.DATASET_NAME == 'DOTA2.0':
                name_label_map['container-crane'] = 16
                name_label_map['airport'] = 17
                name_label_map['helipad'] = 18

        elif self.cfgs.DATASET_NAME == 'coco':
            name_label_map = self.coco_name2abel()
        elif self.cfgs.DATASET_NAME == 'pascal':
            name_label_map = {
                'back_ground': 0,
                'aeroplane': 1,
                'bicycle': 2,
                'bird': 3,
                'boat': 4,
                'bottle': 5,
                'bus': 6,
                'car': 7,
                'cat': 8,
                'chair': 9,
                'cow': 10,
                'diningtable': 11,
                'dog': 12,
                'horse': 13,
                'motorbike': 14,
                'person': 15,
                'pottedplant': 16,
                'sheep': 17,
                'sofa': 18,
                'train': 19,
                'tvmonitor': 20
            }
        elif self.cfgs.DATASET_NAME.startswith('DIOR'):
            name_label_map = {
                'back_ground': 0,
                'airplane': 1,
                'airport': 2,
                'baseballfield': 3,
                'basketballcourt': 4,
                'bridge': 5,
                'chimney': 6,
                'dam': 7,
                'Expressway-Service-area': 8,
                'Expressway-toll-station': 9,
                'golffield': 10,
                'groundtrackfield': 11,
                'harbor': 12,
                'overpass': 13,
                'ship': 14,
                'stadium': 15,
                'storagetank': 16,
                'tenniscourt': 17,
                'trainstation': 18,
                'vehicle': 19,
                'windmill': 20,
                'swimmingpool': 21,
                'soccerballfield': 22,
                'volleyballcourt': 23,
                'roundabout': 24,
                'container-crane': 25,
                'helipad': 26,
                'rugbyfield': 27
            }
        elif self.cfgs.DATASET_NAME == 'bdd100k':
            name_label_map = {
                'back_ground': 0,
                'bus': 1,
                'traffic light': 2,
                'traffic sign': 3,
                'person': 4,
                'bike': 5,
                'truck': 6,
                'motor': 7,
                'car': 8,
                'train': 9,
                'rider': 10
            }
        else:
            name_label_map = {}
            assert 'please set label dict!'
        return name_label_map

    def label2name(self):
        label_name_map = {}
        for name, label in self.name2label().items():
            label_name_map[label] = name
        return label_name_map
