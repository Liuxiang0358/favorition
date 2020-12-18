# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:59:17 2020

@author: user
"""

import os
import cv2
from xml.etree.ElementTree import ElementTree,Element


def read_xml(in_path):
    '''''读取并解析xml文件
       in_path: xml路径
       return: ElementTree'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree

def write_xml(tree, out_path):
    '''''将xml文件写出
       tree: xml树
       out_path: 写出路径'''
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

def if_match(node, kv_map):
    '''''判断某个节点是否包含所有传入参数属性
       node: 节点
       kv_map: 属性及属性值组成的map'''
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True

# ----------------search -----------------
def find_nodes(tree, path):
    '''''查找某个路径匹配的所有节点
       tree: xml树
       path: 节点路径'''
    return tree.findall(path)

def get_node_by_keyvalue(nodelist, kv_map):
    '''''根据属性及属性值定位符合的节点，返回节点
       nodelist: 节点列表
       kv_map: 匹配属性及属性值map'''
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes

# ---------------change ----------------------
def change_node_properties(nodelist, kv_map, is_delete=False):
    '''修改/增加 /删除 节点的属性及属性值
       nodelist: 节点列表
       kv_map:属性及属性值map'''
    for node in nodelist:
        for key in kv_map:
            if is_delete:
                if key in node.attrib:
                    del node.attrib[key]
            else:
                node.set(key, kv_map.get(key))

def change_node_text(nodelist, text, is_add=False, is_delete=False):
    '''''改变/增加/删除一个节点的文本
       nodelist:节点列表
       text : 更新后的文本'''
    for node in nodelist:
        if is_add:
            node.text += text
        elif is_delete:
            node.text = ""
        else:
            node.text = text

def create_node(tag, property_map=None, content=None):
    '''新造一个节点
       tag:节点标签
       property_map:属性及属性值map
       content: 节点闭合标签里的文本内容
       return 新节点'''
    element = Element(tag,content, property_map)
    element.text = None
    return element

def add_child_node(nodelist, element):
    '''''给一个节点添加子节点
       nodelist: 节点列表
       element: 子节点'''
    for node in nodelist:
        node.append(element)

def del_node_by_tagkeyvalue(nodelist, tag, kv_map):
    '''''同过属性及属性值定位一个节点，并删除之
       nodelist: 父节点列表
       tag:子节点标签
       kv_map: 属性及属性值列表'''
    for parent_node in nodelist:
        children = parent_node.getchildren()
        for child in children:
            if child.tag == tag and if_match(child, kv_map):
                parent_node.remove(child)

def get_xml(input_dir):
    xml_path_list = []
    for (root_path,dirname,filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.xml'):
                xml_path = root_path+"/"+filename
                xml_path_list.append(xml_path)
    return  xml_path_list

def make_xml(file):
        xml_file = 'label.xml'
        ################ 1. 读取xml文件  ##########
        root = read_xml(xml_file)
        file = file.split(',')
        data = [[1, 1, 1],
                ["abc",1, 1, 1,1],
                ["ABC",1, 1, 1,1]]
        # 修改图像属性
        
        size = root.find('size')
        size.find('height').text = file[0]
        size.find('width').text = file[1]
        size.find('depth').text = file[2]
#        filename = root.findall('filename')
        root.find('filename').text = file[3]
        # Find annotations.
        Object = root.findall('object')
        for i in range(len(Object)):
            if len(Object) ==2:
                # change_node_text(Object[i].find('name'),str(data[i+1][0]))
                Object[i].find('name').text = str(data[i+1][0])                 # 修改节点名
                Object[i].find('bndbox').find('ymin').text = str(data[i+1][1])  # 修改节点文本
                Object[i].find('bndbox').find('xmin').text = str(data[i+1][2])
                Object[i].find('bndbox').find('ymax').text = str(data[i+1][3])
                Object[i].find('bndbox').find('xmax').text = str(data[i+1][4])

            else:
                # change_node_text(Object[0].find('name'),"new text")
                Object[0].find('name').text = 'Gas_tank'                # 修改节点名
                Object[0].find('bndbox').find('ymin').text = str(file[4])  # 修改节点文本
                Object[0].find('bndbox').find('xmin').text = str(file[5])
                Object[0].find('bndbox').find('ymax').text = str(file[6])
                Object[0].find('bndbox').find('xmax').text = str(file[7])


        ################ 输出到结果文件  ##########
        write_xml(root, 'label/' + file[3].replace('.jpg','.xml'))
if __name__ == "__main__":
    
    # input_dir ="label"
    # xml_path_list = get_xml(input_dir)
    f = open("log_new.txt","r")   #设置文件对象
    data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
    f.close()  
    print(len(data))
    # i = 1
    for file in data:
    #    try :   
               filecopy = file
            #    print(i)
               file = file.split('/')[1].replace('\n','')
               file = file.replace(file.split(',')[3],str(int(file.split(',')[1]) +  int(file.split(',')[3])))
               file = file.replace(file.split(',')[4],str(int(file.split(',')[2]) +  int(file.split(',')[4])))
               name = filecopy.split(',')[0]
            #    print(name)
            #    if  110105026-5789-5885-00001084_nobody-51 in 
               img = cv2.imread(name)
               h, w, c = img.shape
               file = str(h)+','+str(w)+','+str(c)+',' + file
               make_xml(file)
        #       print(file)
    #    except:
            #   print(file + 'is error')
        
