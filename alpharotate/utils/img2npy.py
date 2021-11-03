import cv2
import os
import numpy as np

img_path = '/data/dataset/DOTA/test/images/'
npy_path = '/data/dataset/DOTA/test/npy/'
if not os.path.exists(npy_path):
    os.makedirs(npy_path)
images = os.listdir(img_path)

for i in images:
    img = cv2.imread(os.path.join(img_path, i))
    print(img.shape)
    np.save(os.path.join(npy_path, i.replace('.png', '.npy')), img)
    img_npy = np.load(os.path.join(npy_path, i.replace('.png', '.npy')))
    print(img_npy.shape)
