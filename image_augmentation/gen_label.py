# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:28:50 2020

@author: user
"""
import xml.etree.ElementTree as ET
import os
classes = ['Gas_tank','motor','Pram','Pet','Express_package','water','carry','bag','luggage','head','body']  # 输入缺陷名称，必须与xml标注名称一致
# classes_clear = ['Gas_tank','body','head']
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    
    # in_file = open('mixdata/Annotations/' + image_id)  # 读取xml文件路径
    in_file = open('mixdata/Annotations/' + image_id)  # 读取xml文件路径
    
    # out_file = open('text/'  + image_id.replace('.xml','.txt'), 'w')  # 需要保存的txt格式文件路径
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    # print(h)
    Object = root.findall('object')
    for i in range(len(Object)):
        cls = Object[i].find('name').text
        
        if cls not in classes:  # 检索xml中的缺陷名称
            continue
        # if cls in classes_clear:
        #     continue
        cls_id = classes.index(cls)
        print(cls_id)
        # cls_id = 0
        xmlbox = Object[i].find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        xywh = [int(xmlbox.find('xmin').text),int(xmlbox.find('ymin').text),int(xmlbox.find('xmax').text) - int(xmlbox.find('xmin').text),int(xmlbox.find('ymax').text) - int(xmlbox.find('ymin').text)]
        bb = convert((w, h), b)
        # print(bb)
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
        with open('log_3.txt', 'a+') as f:
            f.write( image_id.replace('.xml','.jpg') + ',' + str(cls_id) + ','+ ",".join([str(a) for a in xywh])  + '\n')
    in_file.close()
    # out_file.close()

# image_ids_train =  os.listdir('mixdata/Annotations/') # 读取xml文件名索引
image_ids_train =  os.listdir('mixdata/Annotations/') # 读取xml文件名索引
print(len(image_ids_train))
for image_id in image_ids_train:
    convert_annotation(image_id)
'''
修改 28 、 59 行的文件名
修改输出的 txt 文件名
'''