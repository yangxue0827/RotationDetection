import math, os
import shutil
from dataloader.dataset.FDDB.txt2xml import WriterXMLFiles
import numpy as np
import cv2
import random
from PIL import Image
import glob
import sys
random.seed(2018)


def convert_coord(coord):
    """
    :param coord:[major_axis_radius, minor_axis_radius, angle, center_x, center_y, detection_score]
    :return:
    """
    x_c = coord[3]
    y_c = coord[4]
    w = coord[0]*2
    h = coord[1]*2
    theta = -(coord[2] / math.pi * 180)
    up_l = (-w/2, h/2)
    down_l = (-w/2, -h/2)
    up_r = (w/2, h/2)
    down_r = (w/2, -h/2)

    theta = -theta

    x1 = math.cos(theta/180*math.pi) * up_l[0] - math.sin(theta/180*math.pi) * up_l[1] + x_c
    y1 = math.sin(theta/180*math.pi) * up_l[0] + math.cos(theta/180*math.pi) * up_l[1] + y_c

    x2 = math.cos(theta / 180 * math.pi) * down_l[0] - math.sin(theta / 180 * math.pi) * down_l[1] + x_c
    y2 = math.sin(theta / 180 * math.pi) * down_l[0] + math.cos(theta / 180 * math.pi) * down_l[1] + y_c

    x3 = math.cos(theta / 180 * math.pi) * up_r[0] - math.sin(theta / 180 * math.pi) * up_r[1] + x_c
    y3 = math.sin(theta / 180 * math.pi) * up_r[0] + math.cos(theta / 180 * math.pi) * up_r[1] + y_c

    x4 = math.cos(theta / 180 * math.pi) * down_r[0] - math.sin(theta / 180 * math.pi) * down_r[1] + x_c
    y4 = math.sin(theta / 180 * math.pi) * down_r[0] + math.cos(theta / 180 * math.pi) * down_r[1] + y_c

    return [x1, y1, x2, y2, x3, y3, x4, y4]


def getFiles(file_dir):
    return [file for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir,file))]


def view_bar(num,total):
    """
    进度条
    :param num:
    :param total:
    :return:
    """
    ret = num / total
    ag = int(ret * 50)
    ab = "\r[%-50s]%3d%%%6d/%-6d" % ('=' * ag, 100*ret, num, total)
    sys.stdout.write(ab)
    sys.stdout.flush()


def generateImageAndXml(image_dir,txt_dir,image_save_dir, xml_save_dir):
    """
    :param image_dir: 图片根目录
    :param txt_dir: text文件根目录
    :return:
    """
    files = getFiles(txt_dir)  # 所有txt
    files = [file for file in files if file.endswith('ellipseList.txt')]  # 过滤出annotation文件
    for file in files:
        print(file)
        txt_path = os.path.join(txt_dir,file)
        with open(txt_path, 'r') as f:
            line = 'init'
            while True:
                line = f.readline()
                if line == '':
                    break
                image_dir_tmp = line.strip()
                # print(">>>{} is being processed...".format(image_dir_tmp))
                new_name = '_'.join(image_dir_tmp.split('/'))
                image_path = os.path.join(image_dir, image_dir_tmp+'.jpg')
                img_shape = cv2.imread(image_path).shape
                h = img_shape[0]
                w = img_shape[1]
                d = img_shape[2]
                shutil.copyfile(image_path,os.path.join(image_save_dir,new_name+ '.jpg'))
                line = f.readline().strip()
                num = int(line)
                box_list = []
                for i in range(num):
                    line = f.readline().strip()
                    data = [float(val) for val in line.split(' ') if val != '']
                    coord = convert_coord(data)
                    box_list.append(coord)
                WriterXMLFiles(new_name+'.xml',xml_save_dir,box_list,w,h,d)
def rotateBox(box_list,rotate_matrix,h,w):
    trans_box_list = []
    for bbx in box_list:
        bbx = [[bbx[0]-w//2,bbx[2]-w//2,bbx[4]-w//2,bbx[6]-w//2],
               [bbx[1]-h//2,bbx[3]-h//2,bbx[5]-h//2,bbx[7]-h//2]]
        trans_box_list.append(bbx)
    if len(trans_box_list) == 0:
        return []
    else:
        res_box_list = []
        for bbx in trans_box_list:
            bbx = np.matmul(rotate_matrix,np.array(bbx))
            bbx = bbx + np.array([
                [w//2,w//2,w//2,w//2],
                [h//2,h//2,h//2,h//2]
            ])
            x_mean = np.mean(bbx[0])
            y_mean = np.mean(bbx[1])
            if 0 < x_mean < w and 0 < y_mean < h:
                bbx = [bbx[0,0],bbx[1,0],bbx[0,1],bbx[1,1],bbx[0,2],bbx[1,2],bbx[0,3],bbx[1,3]]
                res_box_list.append(bbx)
        return res_box_list


def aug_data(image_dir,txt_dir,image_save_dir, xml_save_dir,n):
    """
    :param image_dir:
    :param txt_dir:
    :param image_save_dir:
    :param xml_save_dir:
    :param n:增强次数
    :return:
    """
    files = getFiles(txt_dir)  # 所有txt
    files = [file for file in files if file.endswith('ellipseList.txt')]  # 过滤出annotation文件
    for file in files:
        print(file)
        txt_path = os.path.join(txt_dir, file)
        with open(txt_path, 'r') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                image_dir_tmp = line.strip()
                print(">>>{} is being processed...".format(image_dir_tmp))
                new_name = '_'.join(image_dir_tmp.split('/'))
                image_path = os.path.join(image_dir, image_dir_tmp + '.jpg')
                im = Image.open(image_path)
                (w, h) = im.size
                d = 3
                center = (w//2,h//2)
                line = f.readline().strip()
                num = int(line)
                box_list = []
                for i in range(num):
                    line = f.readline().strip()
                    data = [float(val) for val in line.split(' ') if val != '']
                    coord = convert_coord(data)
                    box_list.append(coord)
                ii = 0
                while ii < n:
                    angle = random.randint(1,359)
                    rotate_matrix = np.array([
                        [np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)],
                        [-np.sin(angle * np.pi / 180), np.cos(angle * np.pi / 180)]
                    ])
                    box_list_new = rotateBox(box_list,rotate_matrix,h,w)
                    if len(box_list_new) == 0:
                        continue
                    ii += 1
                    new_im = im.rotate(angle, center=center)
                    new_im.save(os.path.join(image_save_dir,'{}_{}.jpg'.format(new_name,angle)))
                    WriterXMLFiles('{}_{}.xml'.format(new_name,angle), xml_save_dir, box_list_new, w, h, d)


if __name__ == '__main__':
    image_dir = '/data/yangxue/dataset/FDDB/originalPics'
    txt_dir = '/data/yangxue/dataset/FDDB/FDDB-folds'
    image_save_dir = '/data/yangxue/dataset/FDDB/images'
    xml_save_dir = '/data/yangxue/dataset/FDDB/xml'
    generateImageAndXml(image_dir, txt_dir, image_save_dir, xml_save_dir)
    # image_save_dir = '/data/yangxue/dataset/FDDB/images_aug'
    # xml_save_dir = '/data/yangxue/dataset/FDDB/xml_aug'
    # aug_data(image_dir, txt_dir, image_save_dir, xml_save_dir, 10)