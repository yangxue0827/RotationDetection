import os
from xml.dom.minidom import Document
import numpy as np
import copy
import cv2
import sys
sys.path.append('../../..')

from utils.tools import makedirs
from libs.utils.coordinate_convert import backward_convert


def save_to_xml(save_path, im_height, im_width, objects_axis, label_name):
    im_depth = 0
    object_num = len(objects_axis)
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('VOC2007')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_name = doc.createTextNode('000024.jpg')
    filename.appendChild(filename_name)
    annotation.appendChild(filename)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('The VOC2007 Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)

    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)

    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('322409915'))
    source.appendChild(flickrid)

    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('knautia'))
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
        object_name.appendChild(doc.createTextNode(label_name[int(objects_axis[i][-1])]))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
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


class_list = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
              'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court', 'basketball-court',
              'storage-tank', 'soccer-ball-field',
              'roundabout', 'harbor',
              'swimming-pool', 'helicopter', 'container-crane']


def format_label(txt_list):
    format_data = []
    for i in txt_list:
        if len(i.split(' ')) < 9:
            continue
        format_data.append(
            [float(xy) for xy in i.split(' ')[:8]] + [class_list.index(i.split(' ')[8])]
        )

        if i.split(' ')[8] not in class_list:
            print('warning found a new label :', i.split(' ')[8])
            exit()
    return np.array(format_data)


def clip_image(file_idx, image, boxes_all, width, height, stride_w, stride_h):
    min_pixel = 5
    print(file_idx)
    boxes_all_5 = backward_convert(boxes_all[:, :8], False)
    print(boxes_all[np.logical_or(boxes_all_5[:, 2] <= min_pixel, boxes_all_5[:, 3] <= min_pixel), :])
    boxes_all = boxes_all[np.logical_and(boxes_all_5[:, 2] > min_pixel, boxes_all_5[:, 3] > min_pixel), :]

    if boxes_all.shape[0] > 0:
        shape = image.shape
        for start_h in range(0, shape[0], stride_h):
            for start_w in range(0, shape[1], stride_w):
                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)
                start_h_new = start_h
                start_w_new = start_w
                if start_h + height > shape[0]:
                    start_h_new = shape[0] - height
                if start_w + width > shape[1]:
                    start_w_new = shape[1] - width
                top_left_row = max(start_h_new, 0)
                top_left_col = max(start_w_new, 0)
                bottom_right_row = min(start_h + height, shape[0])
                bottom_right_col = min(start_w + width, shape[1])

                subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]

                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col
                box[:, 4] = boxes[:, 4] - top_left_col
                box[:, 6] = boxes[:, 6] - top_left_col

                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row
                box[:, 5] = boxes[:, 5] - top_left_row
                box[:, 7] = boxes[:, 7] - top_left_row
                box[:, 8] = boxes[:, 8]
                center_y = 0.25 * (box[:, 1] + box[:, 3] + box[:, 5] + box[:, 7])
                center_x = 0.25 * (box[:, 0] + box[:, 2] + box[:, 4] + box[:, 6])

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)
                if len(idx) > 0 and (subImage.shape[0] > 5 and subImage.shape[1] > 5):
                    makedirs(os.path.join(save_dir, 'images'))
                    img = os.path.join(save_dir, 'images',
                                       "%s_%04d_%04d.png" % (file_idx, top_left_row, top_left_col))
                    cv2.imwrite(img, subImage)

                    makedirs(os.path.join(save_dir, 'labeltxt'))
                    xml = os.path.join(save_dir, 'labeltxt',
                                       "%s_%04d_%04d.xml" % (file_idx, top_left_row, top_left_col))
                    save_to_xml(xml, subImage.shape[0], subImage.shape[1], box[idx, :], class_list)


print('class_list', len(class_list))
raw_data = '/data/dataset/DOTA/val/'
raw_images_dir = os.path.join(raw_data, 'images', 'images')
raw_label_dir = os.path.join(raw_data, 'labelTxt', 'labelTxt')

save_dir = '/data/dataset/DOTA1.0/trainval/'

images = [i for i in os.listdir(raw_images_dir) if 'png' in i]
labels = [i for i in os.listdir(raw_label_dir) if 'txt' in i]

print('find image', len(images))
print('find label', len(labels))

min_length = 1e10
max_length = 1

img_h, img_w, stride_h, stride_w = 600, 600, 450, 450

for idx, img in enumerate(images):
    print(idx, 'read image', img)
    img_data = cv2.imread(os.path.join(raw_images_dir, img))

    txt_data = open(os.path.join(raw_label_dir, img.replace('png', 'txt')), 'r').readlines()
    box = format_label(txt_data)

    if box.shape[0] > 0:
        clip_image(img.strip('.png'), img_data, box, img_w, img_h, stride_w, stride_h)
