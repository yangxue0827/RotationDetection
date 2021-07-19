import os
import json
import sys
import numpy as np
from xml.dom.minidom import Document
import xml.dom.minidom

sys.path.append('../../..')

from utils.tools import makedirs
from libs.utils.coordinate_convert import backward_convert, forward_convert


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
    fp = open(os.path.join(path, filename), 'w')
    doc.writexml(fp, indent='\n')
    fp.close()


xml_path = '/data/dataset/HRSID_JPG/xmls'
makedirs(xml_path)

fr = open('/data/dataset/HRSID_JPG/annotations/train2017.json', 'r')
data = json.load(fr)

images = data['images']
data_dict = {}
for i in images:
    data_dict[i['id']] = {}
    data_dict[i['id']]['file_name'] = i['file_name']
    data_dict[i['id']]['height'] = i['height']
    data_dict[i['id']]['width'] = i['width']
    data_dict[i['id']]['bbox'] = []
    data_dict[i['id']]['label'] = []

annotations = data['annotations']

for ann in annotations:
    bbox = ann['segmentation'][0]
    label = 'ship'
    data_dict[ann['image_id']]['bbox'].append(bbox)
    data_dict[ann['image_id']]['label'].append(label)

for k in data_dict.keys():
    bbox = backward_convert(np.array(data_dict[k]['bbox']), False)
    bbox = forward_convert(bbox, False)
    labels = data_dict[k]['label']
    make_xml(filename=data_dict[k]['file_name'].split('.')[0]+'.xml',
             path=xml_path,
             box_list=bbox,
             labels=labels,
             w=data_dict[k]['width'],
             h=data_dict[k]['height'],
             d=3)
