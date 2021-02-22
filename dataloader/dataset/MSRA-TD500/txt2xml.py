import os
from xml.dom.minidom import Document
import xml.dom.minidom
import numpy as np
import cv2
import codecs
import math


def WriterXMLFiles(filename, path, box_list, labels, w, h, d):

    # dict_box[filename]=json_dict[filename]
    doc = xml.dom.minidom.Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)

    foldername = doc.createElement("folder")
    foldername.appendChild(doc.createTextNode("JPEGImages"))
    root.appendChild(foldername)

    nodeFilename = doc.createElement('filename')
    nodeFilename.appendChild(doc.createTextNode(filename))
    root.appendChild(nodeFilename)

    pathname = doc.createElement("path")
    pathname.appendChild(doc.createTextNode("xxxx"))
    root.appendChild(pathname)

    sourcename=doc.createElement("source")

    databasename = doc.createElement("database")
    databasename.appendChild(doc.createTextNode("Unknown"))
    sourcename.appendChild(databasename)

    annotationname = doc.createElement("annotation")
    annotationname.appendChild(doc.createTextNode("xxx"))
    sourcename.appendChild(annotationname)

    imagename = doc.createElement("image")
    imagename.appendChild(doc.createTextNode("xxx"))
    sourcename.appendChild(imagename)

    flickridname = doc.createElement("flickrid")
    flickridname.appendChild(doc.createTextNode("0"))
    sourcename.appendChild(flickridname)

    root.appendChild(sourcename)

    nodesize = doc.createElement('size')
    nodewidth = doc.createElement('width')
    nodewidth.appendChild(doc.createTextNode(str(w)))
    nodesize.appendChild(nodewidth)
    nodeheight = doc.createElement('height')
    nodeheight.appendChild(doc.createTextNode(str(h)))
    nodesize.appendChild(nodeheight)
    nodedepth = doc.createElement('depth')
    nodedepth.appendChild(doc.createTextNode(str(d)))
    nodesize.appendChild(nodedepth)
    root.appendChild(nodesize)

    segname = doc.createElement("segmented")
    segname.appendChild(doc.createTextNode("0"))
    root.appendChild(segname)

    for box, label in zip(box_list, labels):

        nodeobject = doc.createElement('object')
        nodename = doc.createElement('name')
        nodename.appendChild(doc.createTextNode(label))
        nodeobject.appendChild(nodename)
        nodebndbox = doc.createElement('bndbox')
        nodex1 = doc.createElement('x1')
        nodex1.appendChild(doc.createTextNode(str(box[0])))
        nodebndbox.appendChild(nodex1)
        nodey1 = doc.createElement('y1')
        nodey1.appendChild(doc.createTextNode(str(box[1])))
        nodebndbox.appendChild(nodey1)
        nodex2 = doc.createElement('x2')
        nodex2.appendChild(doc.createTextNode(str(box[2])))
        nodebndbox.appendChild(nodex2)
        nodey2 = doc.createElement('y2')
        nodey2.appendChild(doc.createTextNode(str(box[3])))
        nodebndbox.appendChild(nodey2)
        nodex3 = doc.createElement('x3')
        nodex3.appendChild(doc.createTextNode(str(box[4])))
        nodebndbox.appendChild(nodex3)
        nodey3 = doc.createElement('y3')
        nodey3.appendChild(doc.createTextNode(str(box[5])))
        nodebndbox.appendChild(nodey3)
        nodex4 = doc.createElement('x4')
        nodex4.appendChild(doc.createTextNode(str(box[6])))
        nodebndbox.appendChild(nodex4)
        nodey4 = doc.createElement('y4')
        nodey4.appendChild(doc.createTextNode(str(box[7])))
        nodebndbox.appendChild(nodey4)

        # ang = doc.createElement('angle')
        # ang.appendChild(doc.createTextNode(str(angle)))
        # nodebndbox.appendChild(ang)
        nodeobject.appendChild(nodebndbox)
        root.appendChild(nodeobject)
    fp = open(os.path.join(path,filename), 'w')
    doc.writexml(fp, indent='\n')
    fp.close()


def coordinate_convert_r(box):
    w, h = box[2:-1]
    theta = -box[-1]
    x_lu, y_lu = -w/2, h/2
    x_ru, y_ru = w/2, h/2
    x_ld, y_ld = -w/2, -h/2
    x_rd, y_rd = w/2, -h/2

    x_lu_ = math.cos(theta)*x_lu + math.sin(theta)*y_lu + box[0] + w / 2
    y_lu_ = -math.sin(theta)*x_lu + math.cos(theta)*y_lu + box[1] + h / 2

    x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0] + w / 2
    y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1] + h / 2

    x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0] + w / 2
    y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1] + h / 2

    x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0] + w / 2
    y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1] + h / 2

    convert_box = [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]

    return convert_box


def load_annoataion(txt_path):
    boxes, labels = [], []
    fr = codecs.open(txt_path, 'r', 'utf-8')
    lines = fr.readlines()

    for line in lines:
        b = line.strip('\ufeff').strip('\xef\xbb\xbf').strip('$').split(' ')
        line = list(map(float, b))
        boxes.append(coordinate_convert_r(line[2:]))
        labels.append('text')

    return np.array(boxes), np.array(labels)


if __name__ == "__main__":
    txt_path = '/data/dataset/MSRA-TD500/train/labels'
    xml_path = '/data/dataset/MSRA-TD500/train/xmls'
    img_path = '/data/dataset/MSRA-TD500/train/images'
    print(os.path.exists(txt_path))
    txts = os.listdir(txt_path)
    for count, t in enumerate(txts):
        boxes, labels = load_annoataion(os.path.join(txt_path, t))
        # boxes = coordinate_convert_r(boxes[:, 2:])
        xml_name = t.replace('.gt', '.xml')
        img_name = t.replace('.gt', '.JPG')
        img = cv2.imread(os.path.join(img_path, img_name))
        # for b in boxes:
        #     b = np.array(b, np.int32)
        #     img = cv2.line(img, (b[0], b[1]), (b[2], b[3]), thickness=3, color=(0, 255, 255))
        #     img = cv2.line(img, (b[2], b[3]), (b[4], b[5]), thickness=3, color=(0, 255, 255))
        #     img = cv2.line(img, (b[4], b[5]), (b[6], b[7]), thickness=3, color=(0, 255, 255))
        #     img = cv2.line(img, (b[6], b[7]), (b[0], b[1]), thickness=3, color=(0, 255, 255))
        # cv2.imwrite('./test.jpg', img)
        h, w, d = img.shape
        WriterXMLFiles(xml_name, xml_path, boxes, labels, w, h, d)

        if count % 1000 == 0:
            print(count)