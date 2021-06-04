import os
from xml.dom.minidom import Document
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import csv
import cv2
import codecs
import sys

sys.path.append('../../..')


from libs.utils.mask_sample import points_sampling


def make_xml(filename, path, box_list, labels, w, h, d):

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
        for ii in range(box_list.shape[1]//2):
            nodex = doc.createElement('x{}'.format(ii+1))
            nodex.appendChild(doc.createTextNode(str(box[2*ii])))
            nodebndbox.appendChild(nodex)
            nodey = doc.createElement('y{}'.format(ii+1))
            nodey.appendChild(doc.createTextNode(str(box[2*ii+1])))
            nodebndbox.appendChild(nodey)

        nodeobject.appendChild(nodebndbox)
        root.appendChild(nodeobject)
    fp = open(os.path.join(path,filename), 'w')
    doc.writexml(fp, indent='\n')
    fp.close()


def load_annoataion(txt_path):
    boxes, labels = [], []
    fr = codecs.open(txt_path, 'r', 'utf-8')
    lines = fr.readlines()

    for line in lines:
        b = line.split(',')[:-1]
        line = np.array(list(map(int, b)))

        line = points_sampling(line.reshape([-1, 2]), 12)

        boxes.append(line.reshape([-1, ]))
        labels.append('text')

    return np.array(boxes), np.array(labels)


if __name__ == "__main__":
    txt_path = '/mnt/nas/home/yangxue/dataset/Total_Text/labels/train_gts'
    xml_path = '/mnt/nas/home/yangxue/dataset/Total_Text/xmls/train'
    img_path = '/mnt/nas/home/yangxue/dataset/Total_Text/Images/Train'
    print(os.path.exists(txt_path))
    imgs = os.listdir(img_path)
    for count, i in enumerate(imgs):
        t = os.path.join(txt_path, i+'.txt')
        boxes, labels = load_annoataion(t)
        x = i.split('.')[0] + '.xml'

        img = cv2.imread(os.path.join(img_path, i))

        if img is None:
            print(i)
            continue

        if i.split('.')[-1] in ['png', 'bmp']:
            print(i)
            cv2.imwrite(os.path.join(img_path, i.split('.')[0] + '.jpg'), img)

        h, w, d = img.shape
        make_xml(x, xml_path, boxes, labels, w, h, d)

        if count % 1000 == 0:
            print(count)
