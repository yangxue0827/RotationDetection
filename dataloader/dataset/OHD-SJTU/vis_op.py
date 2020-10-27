import os
import cv2
import numpy as np


img_path = '/data/yangxue/dataset/OHD-SJTU/all_data/images'
images_vis_path = '/data/yangxue/dataset/OHD-SJTU/all_data/images_vis'
txt_path = '/data/yangxue/dataset/OHD-SJTU/all_data/polygon_txt'


all_txt = os.listdir(txt_path)


for t in all_txt:

    img = cv2.imread(os.path.join(img_path, t.replace('txt', 'jpg')))
    fr = open(os.path.join(txt_path, t), 'r')
    data = fr.readlines()
    fr.close()
    print(len(data))

    for d in data:
        dd = [int(float(xy)) for xy in d.split(' ')[:-1]]

        if d.split(' ')[-1] == 'ship\n':
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        dd_ = np.array(dd).reshape(-1, 2)
        cv2.polylines(img, [dd_], thickness=3, color=color, isClosed=True)
        cv2.line(img, (dd[0], dd[1]), (dd[0], dd[1]), thickness=10, color=(0, 0, 255))
    cv2.imwrite(os.path.join(images_vis_path, t.replace('txt', 'jpg')), img)
