# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:28:50 2020

@author: liuxiang
"""

import cv2
import numpy as np
import os
import uuid
import random
import xml.etree.ElementTree as ET
classes = ['Gas_tank','motor','Pram','Pet','Express_package','water','carry','bag','luggage','head','body']

def Contrast_and_Brightness1(img ,alpha):
    '''
    改变图像亮度
    alpha 为改变系数 
    '''
    beta = 1
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
    return dst

def Contrast_and_Brightness(img ):
    alpha = 0.6 + np.random.random() / 6 
    beta = 1
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
    return dst


def Gaussian(img):
    '''
    5*5 高斯滤波
    '''
    return cv2.GaussianBlur(img, (5, 5), 0)

def isgary(img):
    '''
    判断是否为灰度图像
    inpput: img
    '''
       img = np.transpose(img,(2,0,1))
       mean1 = np.mean(img[0,:,:])
       mean2 = np.mean(img[1,:,:])
       mean3 = np.mean(img[1,:,:])
       if mean1 == mean2 == mean3:
              return True
       else:
              return False

def rotate(image):  # 1
    '''
    对图像进行旋转，旋转角度为[-20,20]
    input：img
    '''
    (h, w) = image.shape[:2]  # 2
    angle = np.random.random() * np.random.choice([-10, 10]) * 2
    center = (w // 2, h // 2)  # 4

    M = cv2.getRotationMatrix2D(center, angle, 1)  # 5

    rotated = cv2.warpAffine(image, M, (w, h))  # 6
    return rotated

def rad(x):
  '''
  将角度换算成弧度
  input: x (角度)
  output ： 弧度
  '''
  return x * np.pi / 180

    def gray2bgr(img):
   '''
   将单通道的灰度图像转换成三通道的图像
   input： gray img
   output ： color img
   '''
      return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

def convert(size, box):
    '''
    label 转换
    input :x1,y1,x2,y2 (顶点)
    output：x,y,w,h   (中心点)
    '''
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
# 扩展图像，保证内容不超出可视范围
def rotate_3d(img):
    '''
    对图像进行透视变换
    input： img
    output： img
    '''
         w, h = img.shape[0:2]
       
         anglex = np.random.randint(-45,45)
         angley = np.random.randint(-60,60)
         anglez = np.random.choice([-90,90])
#         angley = 0
         anglez = 0
         fov = 42

         z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
          # 齐次变换矩阵
         rx = np.array([[1, 0, 0, 0],
          [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
          [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
          [0, 0, 0, 1]], np.float32)
       
         ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
          [0, 1, 0, 0],
          [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
          [0, 0, 0, 1]], np.float32)
       
         rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
          [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]], np.float32)
       
         r = rx.dot(ry).dot(rz)
       
         # 四对点的生成
         pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
       
         p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
         p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
         p3 = np.array([0, h, 0, 0], np.float32) - pcenter
         p4 = np.array([w, h, 0, 0], np.float32) - pcenter
       
         dst1 = r.dot(p1)
         dst2 = r.dot(p2)
         dst3 = r.dot(p3)
         dst4 = r.dot(p4)
       
         list_dst = [dst1, dst2, dst3, dst4]
       
         org = np.array([[0, 0],
          [w, 0],
          [0, h],
          [w, h]], np.float32)
       
         dst = np.zeros((4, 2), np.float32)
       
         # 投影至成像平面
         for i in range(4):
           dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
           dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
       
         warpR = cv2.getPerspectiveTransform(org, dst)
       
         result = cv2.warpPerspective(img, warpR, (h, w))
       
         return result

def flip(img,flipcode):
    '''
    对图像进行镜像变换
    input：img  
           flipcode  [-1,0,1]
    output:img
    
    '''
    return cv2.flip(img,flipcode)


def reshape(img):
    '''
    对图像的长宽比进行改变
    '''
    return cv2.resize(img, dsize=None, fx=(1 - np.random.random() / 10), fy=(1 - np.random.random() / 10),
                      interpolation=cv2.INTER_LINEAR)

def convert_annotation(image_id):
    '''
    将xml的标签转换成txt  （x1，y1,w,h） 顶点坐标
    imput ： xml文件索引
    '''
    # in_file = open('mixdata/Annotations/' + image_id)  # 读取xml文件路径
    in_file = open(image_id)  # 读取xml文件路径
    label_list = []
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
        # print(cls_id)
        # cls_id = 0
        xmlbox = Object[i].find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        xywh = [int(xmlbox.find('xmin').text),int(xmlbox.find('ymin').text),int(xmlbox.find('xmax').text) - int(xmlbox.find('xmin').text),int(xmlbox.find('ymax').text) - int(xmlbox.find('ymin').text)]
        bb = convert((w, h), b)
        # print(bb)
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'

        label_list.append(',' + str(cls_id) + ','+ ",".join([str(a) for a in xywh])  + '\n')
    in_file.close()
    return label_list

def synthesis(img1_name, img2_name_list, path_result_save,log_save):
    '''
    对图像进行合成 
    input: 
        img1_name : 需要合成的背景照片 ，<name>.jpg
        img2_name_list :  type ：list     [<name1>.jpg,<name2>.jpg]
        path_result_save : 输出图像保存路径
        log_save ： 输出标签保存路径
    output: None
    '''
    thresh = 5
    img1 = cv2.imread(img1_name)
    h, w = img1.shape[:-1]

    for n,img2_name in enumerate(img2_name_list):
        img2 = cv2.imread(img2_name)
        
        h2, w2 = img2.shape[:-1]

        img2 = rotate_3d(img2)

        img2 = cv2.resize(img2,(w2,h2))
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if isgary(img1):
            img2 = gray2bgr(img2gray)
        ret, mask = cv2.threshold(img2gray, thresh, 1, cv2.THRESH_BINARY)
        # print(mask)
        hh = np.sum(mask, axis=1)  ####  h
        # mask[mask == 255] = 1
        location = np.where(hh > 2)
        hh1, hh2 = np.min(location), np.max(location)
        # print(location)
        ww = np.sum(mask, axis=0)  ####  h
        location = np.where(ww > 2)
        ww1, ww2 = np.min(location), np.max(location)

        img2 = img2[hh1:hh2, ww1:ww2, :]

        print(img2.shape) 
        dsizes = (int(w * random.uniform(0.13,0.15)),int(h*random.uniform(0.15,0.18)))
        # print(dsizes)
        # dsizes = (128,268)
        img2 = cv2.resize(img2,dsizes)
    
        img2 = Contrast_and_Brightness1(img2,0.6)
    

        rand = np.random.random()
        if rand <= 0.2:
            img2 = reshape(img2)
        elif rand <= 0.4:
            img2 = flip(img2,np.random.choice([-1,0,1]))
        elif rand <= 0.6:
            img2 = rotate(img2)
        elif rand <= 0.8:
            img2 = Gaussian(img2)
        else:
            img2 = Contrast_and_Brightness(img2)

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)/
        # mean1 = np.mean(gray1).astype(np.uint8)
        # length = len(np.nonzero(gray2)[0])
        # mean2 = np.uint8(np.sum(gray2) / length)
        
        # wrap = np.random.choice([0,1,2,3])
        # h2,w2 = img2.shape[:-1]
        # if wrap == 0:
        #     w_rand = random.randint(0,np.int(0.1*w2))
        #     img2 = img2[w_rand:w2,:]
        # elif wrap == 1:
        #     w_rand = random.randint(0,np.int(0.1*w2))
        #     img2 = img2[0:w2-w_rand,:] 
        # elif wrap == 2:
        #     h_rand = random.randint(0,np.int(0.1*h2))
        #     img2 = img2[:,h_rand:h2] 
        # else:
        #     h_rand = random.randint(0,np.int(0.1*h2))
        #     img2 = img2[:,:h2 - h_rand] 

        rows, cols, channels = img2.shape
        # if 'new' in img2_name:
        #     y_hold =np.int(0.66 * h)
        # else:
        #     y_hold = 0
        if n == 0:
            y_hold = 0
            y = random.randint( y_hold, np.int(0.33 * h)- rows)
        elif n == 1:
            # y_hold = 0
            y = random.randint( np.int(0.33 * h),np.int(0.66 * h)- rows)
        else:
            # y_hold = 0
            y = random.randint( np.int(0.33 * h), h - rows)



        x = random.randint( 0, w - cols )
        # print(x,y)
        roi = img1[y:y + rows, x:x + cols]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, thresh, 255, cv2.THRESH_BINARY)
        mask = ~mask
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        img1[y:y + rows, x:x + cols] = dst

    #    cv2.imshow('res',img1)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
        img2_name = '-'.join([i.split('/')[-1][:-4] for i in img2_name_list])
        save_log = img1_name.split('/')[-1][:-4] + '-' + img2_name + '.jpg' + ',' + '0'+ ','+ str(x) + ',' + str(y) + ',' + str(int(cols)) + ',' + str(int(rows)) + '\n'
        with open(log_save, 'a+') as f:
             f.write(save_log)
    img2_name = '-'.join([i.split('/')[-1][:-4] for i in img2_name_list])
    # print(img2_name)
    save = path_result_save + '/' + img1_name.split('/')[-1][:-4] + '-' + img2_name + '.jpg'
    cv2.imwrite(save, img1)
    
    # print(save , '  ',dsizes)
    # print(x,y)
    
    xml_path = image1_name.replace('.jpg','.xml')
    # print(xml_path)
    if os.path.exists(xml_path):
        with open(log_save, 'a+') as f:
            label_list = convert_annotation(xml_path)
            for label in label_list:
                label = img1_name.split('/')[-1][:-4] + '-' + img2_name + '.jpg' + label
                f.write(label)


if __name__ == '__main__':
    path_Elevator = 'nobody_deal'       ##  电梯图片路径
    # path_Elevator = 'result1'     
    path_gas = 'img'                   ## 需要合成的图片路径
    # path_Elevator = 'mixdata/JPEGImages'
    path_result_save = 'test'     ##  保存合成图片的路径
    log_save = 'log_test.txt'     ##  
    file_gas = os.listdir(path_gas)
    file_floor = os.listdir(path_Elevator)
    file_floor = [file   for file in file_floor if '.xml' not in  file]
    print(len(file_floor))
    # random.seed()
    img2_name_list = []
    # enumerate
    random.shuffle(file_gas)
    for k,f_elsevator in  enumerate(file_floor[:1]):
            print(k)
        # if '110102015-6904-1738_E16989307_1592643644927_10000_4.jpg' ==  f_elsevator:
            random.shuffle(file_gas)
            for i in range(60):
                # try:
                    image1_name = os.path.join(path_Elevator, f_elsevator)
                    f_gas = file_gas[i]
                    image2_name = os.path.join(path_gas, f_gas)
                    # print(image1_name,f_gas)
                    # image_shape = cv2.imread(image1_name).shape
                    # if 'nobody' in image1_name:\
                    img2_name_list.append(image2_name)
                    if (i+1) % 3 == 0:
                        synthesis(image1_name, img2_name_list, path_result_save,log_save)
                        img2_name_list = []
                    # else:
                        # print(image1_name)
                # except:
                    # print(f_gas ,'  is error')
