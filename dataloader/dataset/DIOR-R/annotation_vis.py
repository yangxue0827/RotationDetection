import cv2
import os
import numpy as np
import json
import glob
import shutil

from libs.utils.draw_box_in_img import DrawBox
from libs.configs import cfgs


def load_label(json_dir):
    fr = open(json_dir, 'r')
    data = json.load(fr)

    boxes = []

    for d in data['shapes']:

        points = np.array(d['points'], np.float32)
        hull = cv2.convexHull(np.reshape(points, [-1, 2]))
        rect = cv2.minAreaRect(hull)

        x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]

        if theta == 0:
            w, h = h, w
            theta -= 90
        try:
            boxes.append([x, y, w, h, theta, int(d['label'])])
        except:
            print('wrong label', json_dir)
        if int(d['label']) not in range(26):
            print(json_dir)
        #     if os.path.exists(json_dir):
        #         os.remove(json_dir)
        #     continue
        # if int(d['label']) in [6, 15, 1]:
        #     if os.path.exists(json_dir):
        #         os.remove(json_dir)

    fr.close()

    return np.array(boxes)

jsons = glob.glob('/Users/yangxue/Desktop/JPEGImages-trainval-Draw/*.json')
for j in jsons:
    gt = load_label(j)
    img = cv2.imread(j.replace('.json', '.jpg'))
    drawer = DrawBox(cfgs)
    final_detections = drawer.draw_boxes_with_label_and_scores(np.array(img, np.float32),
                                                               boxes=gt,
                                                               labels=np.ones_like(gt[:, -1]),
                                                               scores=np.ones_like(gt[:, -1]),
                                                               method=1,
                                                               in_graph=False)
    cv2.imwrite(os.path.join('/Users/yangxue/Desktop/tmp', j.split('/')[-1].replace('.json', '.jpg')), final_detections)

