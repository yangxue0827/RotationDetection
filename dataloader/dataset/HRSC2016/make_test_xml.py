import xml.etree.cElementTree as ET
from xml.dom.minidom import Document
import xml.dom.minidom
import numpy as np
import os
import math
import sys
sys.path.append('../../..')

from libs.label_name_dict.label_dict import LABEL_NAME_MAP


def coordinate_convert_r(box):
    w, h = box[2:-1]
    theta = -box[-1]
    x_lu, y_lu = -w/2, h/2
    x_ru, y_ru = w/2, h/2
    x_ld, y_ld = -w/2, -h/2
    x_rd, y_rd = w/2, -h/2

    x_lu_ = math.cos(theta)*x_lu + math.sin(theta)*y_lu + box[0]
    y_lu_ = -math.sin(theta)*x_lu + math.cos(theta)*y_lu + box[1]

    x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
    y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

    x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
    y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

    x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
    y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

    convert_box = [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]

    return convert_box


def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        # if child_of_root.tag == 'size':
        #     for child_item in child_of_root:
        #         if child_item.tag == 'width':
        #             img_width = int(child_item.text)
        #         if child_item.tag == 'height':
        #             img_height = int(child_item.text)
        #
        # if child_of_root.tag == 'object':
        #     label = None
        #     for child_item in child_of_root:
        #         if child_item.tag == 'name':
        #             label = NAME_LABEL_MAP[child_item.text]
        #         if child_item.tag == 'bndbox':
        #             tmp_box = []
        #             for node in child_item:
        #                 tmp_box.append(float(node.text))
        #             assert label is not None, 'label is none, error'
        #             tmp_box.append(label)
        #             box_list.append(tmp_box)

        # ship
        if child_of_root.tag == 'Img_SizeWidth':
            img_width = int(child_of_root.text)
        if child_of_root.tag == 'Img_SizeHeight':
            img_height = int(child_of_root.text)
        if child_of_root.tag == 'HRSC_Objects':
            box_list = []
            for child_item in child_of_root:
                if child_item.tag == 'HRSC_Object':
                    label = 1
                    # for child_object in child_item:
                    #     if child_object.tag == 'Class_ID':
                    #         label = NAME_LABEL_MAP[child_object.text]
                    tmp_box = [0., 0., 0., 0., 0.]
                    for node in child_item:
                        if node.tag == 'mbox_cx':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'mbox_cy':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'mbox_w':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'mbox_h':
                            tmp_box[3] = float(node.text)
                        if node.tag == 'mbox_ang':
                            tmp_box[4] = float(node.text)

                    tmp_box = coordinate_convert_r(tmp_box)
                        # assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    # if len(tmp_box) != 0:
                    box_list.append(tmp_box)
            # box_list = coordinate_convert(box_list)
            # print(box_list)
    gtbox_label = np.array(box_list, dtype=np.int32)

    return img_height, img_width, gtbox_label


def WriterXMLFiles(filename, path, gtbox_label_list, w, h, d):

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

    for gtbox_label in gtbox_label_list:

        nodeobject = doc.createElement('object')
        nodename = doc.createElement('name')
        nodename.appendChild(doc.createTextNode(str(LABEL_NAME_MAP[gtbox_label[-1]])))
        nodeobject.appendChild(nodename)

        nodetruncated = doc.createElement('truncated')
        nodetruncated.appendChild(doc.createTextNode(str(0)))
        nodeobject.appendChild(nodetruncated)

        nodedifficult = doc.createElement('difficult')
        nodedifficult.appendChild(doc.createTextNode(str(0)))
        nodeobject.appendChild(nodedifficult)

        nodepose = doc.createElement('pose')
        nodepose.appendChild(doc.createTextNode('xxx'))
        nodeobject.appendChild(nodepose)

        nodebndbox = doc.createElement('bndbox')
        nodex1 = doc.createElement('x1')
        nodex1.appendChild(doc.createTextNode(str(gtbox_label[0])))
        nodebndbox.appendChild(nodex1)
        nodey1 = doc.createElement('y1')
        nodey1.appendChild(doc.createTextNode(str(gtbox_label[1])))
        nodebndbox.appendChild(nodey1)
        nodex2 = doc.createElement('x2')
        nodex2.appendChild(doc.createTextNode(str(gtbox_label[2])))
        nodebndbox.appendChild(nodex2)
        nodey2 = doc.createElement('y2')
        nodey2.appendChild(doc.createTextNode(str(gtbox_label[3])))
        nodebndbox.appendChild(nodey2)
        nodex3 = doc.createElement('x3')
        nodex3.appendChild(doc.createTextNode(str(gtbox_label[4])))
        nodebndbox.appendChild(nodex3)
        nodey3 = doc.createElement('y3')
        nodey3.appendChild(doc.createTextNode(str(gtbox_label[5])))
        nodebndbox.appendChild(nodey3)
        nodex4 = doc.createElement('x4')
        nodex4.appendChild(doc.createTextNode(str(gtbox_label[6])))
        nodebndbox.appendChild(nodex4)
        nodey4 = doc.createElement('y4')
        nodey4.appendChild(doc.createTextNode(str(gtbox_label[7])))
        nodebndbox.appendChild(nodey4)

        # ang = doc.createElement('angle')
        # ang.appendChild(doc.createTextNode(str(angle)))
        # nodebndbox.appendChild(ang)
        nodeobject.appendChild(nodebndbox)
        root.appendChild(nodeobject)
    fp = open(os.path.join(path, filename), 'w')
    doc.writexml(fp, indent='\n')
    fp.close()


if __name__ == '__main__':
    src_xml_path = '/data/HRSC2016/HRSC2016/Test/Annotations'
    xml_path = '/data/HRSC2016/HRSC2016/Test/xmls'

    src_xmls = os.listdir(src_xml_path)

    for x in src_xmls:
        x_path = os.path.join(src_xml_path, x)
        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(x_path)
        WriterXMLFiles(x, xml_path, gtbox_label, img_width, img_height, 3)
