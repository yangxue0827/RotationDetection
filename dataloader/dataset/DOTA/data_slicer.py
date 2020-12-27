import os
from xml.dom.minidom import Document
import numpy as np
import copy
import cv2
import sys
sys.path.append('../../..')

from utils.tools import makedirs
from libs.utils.coordinate_convert import backward_convert


class_list = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
              'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court', 'basketball-court',
              'storage-tank', 'soccer-ball-field',
              'roundabout', 'harbor',
              'swimming-pool', 'helicopter', 'container-crane']


def save_to_xml(save_path, im_height, im_width, objects_axis, label_name):
    im_depth = 0
    object_num = len(objects_axis)
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('DOTA1.0')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_name = doc.createTextNode(save_path.split('/')[-1].split('.')[0])
    filename.appendChild(filename_name)
    annotation.appendChild(filename)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('The DOTA Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('XML'))
    source.appendChild(annotation_s)

    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)

    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('xxxxxxxx'))
    source.appendChild(flickrid)

    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('xxxxxxxx'))
    owner.appendChild(flickrid_o)

    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('yang'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(im_width)))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(im_height)))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(im_depth)))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(object_num):
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(label_name[int(objects_axis[i][8])]))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode(str(objects_axis[i][8])))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        x0 = doc.createElement('x0')
        x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
        bndbox.appendChild(x0)
        y0 = doc.createElement('y0')
        y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
        bndbox.appendChild(y0)

        x1 = doc.createElement('x1')
        x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
        bndbox.appendChild(x1)
        y1 = doc.createElement('y1')
        y1.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
        bndbox.appendChild(y1)

        x2 = doc.createElement('x2')
        x2.appendChild(doc.createTextNode(str((objects_axis[i][4]))))
        bndbox.appendChild(x2)
        y2 = doc.createElement('y2')
        y2.appendChild(doc.createTextNode(str((objects_axis[i][5]))))
        bndbox.appendChild(y2)

        x3 = doc.createElement('x3')
        x3.appendChild(doc.createTextNode(str((objects_axis[i][6]))))
        bndbox.appendChild(x3)
        y3 = doc.createElement('y3')
        y3.appendChild(doc.createTextNode(str((objects_axis[i][7]))))
        bndbox.appendChild(y3)

    f = open(save_path, 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()


def read_data(lines):
    all_data = []
    for i in lines:
        if len(i.split(' ')) < 10:
            continue
        all_data.append(
            [float(xy) for xy in i.split(' ')[:8]] + [class_list.index(i.split(' ')[8])] + [int(class_list.index(i.split(' ')[9]))]
        )

        if i.split(' ')[8] not in class_list:
            print('warning found a new label :', i.split(' ')[8])
            exit()
    return np.array(all_data)


def clip_image(file_idx, image, boxes_all, width, height, w_overlap, h_overlap):
    print(file_idx)

    # fill useless boxes
    min_pixel = 5
    boxes_all_5 = backward_convert(boxes_all[:, :8], False)
    small_boxes = boxes_all[np.logical_or(boxes_all_5[:, 2] <= min_pixel, boxes_all_5[:, 3] <= min_pixel), :]
    cv2.fillConvexPoly(image, np.reshape(small_boxes, [-1, 2]), color=(0, 0, 0))
    different_boxes = boxes_all[boxes_all[:, 9] == 1]
    cv2.fillConvexPoly(image, np.reshape(different_boxes, [-1, 2]), color=(0, 0, 0))

    boxes_all = boxes_all[np.logical_and(boxes_all_5[:, 2] > min_pixel, boxes_all_5[:, 3] > min_pixel), :]
    boxes_all = boxes_all[boxes_all[:, 9] == 0]

    if boxes_all.shape[0] > 0:

        imgH = image.shape[0]
        imgW = image.shape[1]

        if imgH < height:
            temp = np.zeros([height, imgW, 3], np.float32)
            temp[0:imgH, :, :] = image
            image = temp
            imgH = height

        if imgW < width:
            temp = np.zeros([imgH, width, 3], np.float32)
            temp[:, 0:imgW, :] = image
            image = temp
            imgW = width

        for hh in range(0, imgH, height - h_overlap):
            if imgH - hh - 1 < height:
                hh_ = imgH - height
            else:
                hh_ = hh
            for ww in range(0, imgW, width - w_overlap):
                if imgW - ww - 1 < width:
                    ww_ = imgW - width
                else:
                    ww_ = ww
                subimg = image[hh_:(hh_ + height), ww_:(ww_ + width), :]

                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)

                top_left_row = max(hh_, 0)
                top_left_col = max(ww_, 0)
                bottom_right_row = min(hh_ + height, imgH)
                bottom_right_col = min(ww_ + width, imgW)

                box[:, :8:2] = boxes[:, :8:2] - top_left_col
                box[:, 1:8:2] = boxes[:, 1:8:2] - top_left_row
                box[:, 8:] = boxes[:, 8:]
                center_y = 0.25 * (box[:, 1] + box[:, 3] + box[:, 5] + box[:, 7])
                center_x = 0.25 * (box[:, 0] + box[:, 2] + box[:, 4] + box[:, 6])

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)
                if len(idx) > 0:

                    makedirs(os.path.join(save_dir, 'images'))
                    img = os.path.join(save_dir, 'images',
                                       "%s_%04d_%04d.png" % (file_idx, top_left_row, top_left_col))
                    cv2.imwrite(img, subimg)

                    makedirs(os.path.join(save_dir, 'labeltxt'))
                    xml = os.path.join(save_dir, 'labeltxt',
                                       "%s_%04d_%04d.xml" % (file_idx, top_left_row, top_left_col))

                    save_to_xml(xml, subimg.shape[0], subimg.shape[1], box[idx, :], class_list)


if __name__ == '__main__':
    print('class_list', len(class_list))
    raw_data = '/data/yangxue/dataset/DOTA/val/'
    raw_images_dir = os.path.join(raw_data, 'images', 'images')
    raw_label_dir = os.path.join(raw_data, 'labelTxt', 'labelTxt')

    save_dir = '/data/yangxue/dataset/DOTA/DOTA/trainval/'

    images = [i for i in os.listdir(raw_images_dir) if 'png' in i]
    labels = [i for i in os.listdir(raw_label_dir) if 'txt' in i]

    print('find image', len(images))
    print('find label', len(labels))

    min_length = 1e10
    max_length = 1

    img_h, img_w, h_overlap, w_overlap = 600, 600, 150, 150

    for idx, img in enumerate(images):
        print(idx, 'read image', img)
        img_data = cv2.imread(os.path.join(raw_images_dir, img))

        txt_data = open(os.path.join(raw_label_dir, img.replace('png', 'txt')), 'r').readlines()
        box = read_data(txt_data)

        if box.shape[0] > 0:
            clip_image(img.strip('.png'), img_data, box, img_w, img_h, w_overlap, h_overlap)
