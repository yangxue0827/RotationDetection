# -*- coding:utf-8 -*-
import os
import sys

from multiprocessing import Pool
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 200000000


process_num = 64
ia.seed(1)


def preprocess_handler(img_name, img_dir, rot_list, out_img_dir='/data/dataset/SKU110K/SKU110K-R/images'):
    img_path = os.path.join(img_dir, img_name)
    try:
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
    except:
        try:
            img = cv2.imread(img_path)
        except:
            print(img_path)

    for ang in rot_list:
        seq = iaa.Sequential([
            iaa.Affine(
                rotate=ang,
                fit_output=True
            )
        ])

        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([img])[0]
        out_img_name = 'rotate_aug_{}_'.format(str(ang))
        out_img_name = out_img_name + img_name
        if out_img_dir is None:
            out_dir = os.path.join(img_dir, out_img_name)
        else:
            out_dir = os.path.join(out_img_dir, out_img_name)
        cv2.imwrite(out_dir, image_aug, [int(cv2.IMWRITE_JPEG_QUALITY), 81])


def main(img_dir):
    rotate_angle_list = [-45, -30, -15, 15, 30, 45]
    p = Pool(process_num)
    for img_name in os.listdir(img_dir):
        p.apply_async(preprocess_handler, args=(img_name, img_dir, rotate_angle_list))
    p.close()
    p.join()

if __name__ == '__main__':
    root_img_dir = sys.argv[1]
    main(root_img_dir)