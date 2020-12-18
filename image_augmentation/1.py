'''
先删除 txt  和 result 里面的残余文件 
'''

import os
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
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

def xywh2x1y1w1h1_yolov3(logpath,train_save,val_save,label_save):
    '''
    转化为 yolov3 坐标
    xywh 为顶点坐标，为像素值
    x1y1w1h1 为归一化坐标
    '''
    f = open(logpath,"r")   #设置文件对象
    data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
    f.close()  
    # data = data[:80000]  
    # ?+   data[72375:]         #   原来的数据格式为  72375  ，
    print(len(data))
    label = []
    filelist = []
    for file in data:
    #    try :   
                filecopy = file
            #    print(i)
                file = file.replace('\n','')
                bbox  = file.split(',')
                kinds = bbox[1]
                bbox[4] = int(file.split(',')[2]) +  int(file.split(',')[4])
                bbox[5] = int(file.split(',')[3]) +  int(file.split(',')[5])
                # name = filecopy.split(',')[0]
                name = filecopy.split(',')[0]
                box = (int(bbox[2]),bbox[4],int(bbox[3]),bbox[5])
                # print('old is :',file)
                # print('new is :',box)
                # print(box)
            #    if  110105026-5789-5885-00001084_nobody-51 in 
                # img = cv2.imread(name)
                img = Image.open(os.path.join('result_3',name))
                # print(img.size)
                w, h = img.size
                text = convert((w,h), box)
                Flag = True
                for i in text:
                    if i <=0:
                        Flag = False
                if Flag == False:
                    print(name  , '        error')
                    continue
                # print(text)
                
                filename = name
                label.append([filename,text,kinds])
                filelist.append('data/custom/images/'+filename )
                # print([filename,text],'\n')
                # print('data/custom/images/'+filename,'\n')
                
    filelist = list(set(filelist))
    train_X, test_X = train_test_split(filelist, test_size=0.05, random_state=0,shuffle=True)
    # print(train_X)
    # print(test_X)
    
    train_file = open(train_save, 'w')
    for x in train_X:
        x =  x.replace('.txt','.jpg')
        # print(x)
        train_file.write(x+'\n')
    train_file.close()
    test_file = open(val_save, 'w')
    for x in test_X:
        x = x.replace('.txt','.jpg')
        test_file.write(x+'\n')
    test_file.close()

    for bb in   label:
        with open(os.path.join(label_save,bb[0].replace('.jpg','.txt')),'a+') as f:
            x = bb[2] + ' ' + " ".join([str(a) for a in bb[1]]) + '\n'
            f.write(x)

def isexit1(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder

def isexit2(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)  # make new output folder

def xywh2x1y1w1h1_yolov5(logpath,img_sour,data_path):
    '''
    转化为 yolov3 坐标
    xywh 为顶点坐标，为像素值
    x1y1w1h1 为归一化坐标
    '''
    isexit1(data_path)
    path_images = os.path.join(data_path,'images')
    path_labels = os.path.join(data_path,'labels')
    isexit2(path_images)
    isexit2(path_labels)
    f = open(logpath,"r")   #设置文件对象
    data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
    f.close()  
    # data = data[:80000]  
    # ?+   data[72375:]         #   原来的数据格式为  72375  ，
    print(len(data))
    label = {}
    filelist = []
    for file in data:
    #    try :   
                filecopy = file
            #    print(i)
                file = file.replace('\n','')
                bbox  = file.split(',')
                kinds = bbox[1]
                bbox[4] = int(file.split(',')[2]) +  int(file.split(',')[4])
                bbox[5] = int(file.split(',')[3]) +  int(file.split(',')[5])
                # name = filecopy.split(',')[0]
                name = filecopy.split(',')[0]
                box = (int(bbox[2]),bbox[4],int(bbox[3]),bbox[5])
                # print('old is :',file)
                # print('new is :',box)
                # print(box)
            #    if  110105026-5789-5885-00001084_nobody-51 in 
                # img = cv2.imread(name)
                img = Image.open(os.path.join(img_sour,name))
                # print(img.size)
                w, h = img.size
                text = convert((w,h), box)
                Flag = True
                for i in text:
                    if i <=0:
                        Flag = False
                if Flag == False:
                    print(name  , '        error')
                    continue
                # print(text)
                
                filename = name
                if filename not in label.keys():
                    label[filename] = []
                label[filename].append([text,kinds])
                filelist.append(filename )
                # print([filename,text],'\n')
                # print('data/custom/images/'+filename,'\n')
                
    filelist = list(set(filelist))
    train_X, val_X = train_test_split(filelist, test_size=0.05, random_state=0,shuffle=True)
    # print(train_X)
    # print(test_X)
    path_write = ['train','val']
    isexit2(os.path.join(path_images,path_write[0]))
    isexit2(os.path.join(path_images,path_write[1]))
    isexit2(os.path.join(path_labels,path_write[0]))
    isexit2(os.path.join(path_labels,path_write[1]))
    
    for x in train_X:
        x_txt = x.replace('.jpg','.txt')
        x_jpg =  x
        # print(x_,x_jpg)
        # x_jpg =  x_jpg
        # print(x)
        shutil.copy(os.path.join(img_sour,x_jpg),os.path.join(path_images,path_write[0],x_jpg))
        with open(os.path.join(path_labels,path_write[0],x_txt), 'a+') as f:
            for l in  label[x]:
                 f.write(l[1] + ' ' + " ".join([str(a) for a in l[0]]) + '\n')

    for x in val_X:
        x_txt = x.replace('.jpg','.txt')
        x_jpg =  x
        # print(x)
        # x_jpg =  x_jpg
        shutil.copy(os.path.join(img_sour,x_jpg),os.path.join(path_images,path_write[1],x_jpg))
        with  open(os.path.join(path_labels,path_write[1],x_txt), 'a+') as f:
            for l in  label[x]:
                 f.write(l[1] + ' ' + " ".join([str(a) for a in l[0]]) + '\n')

if __name__ == '__main__':
    '''
    生成yolov3训练数据
    logpath = "log_3.txt"
    train_save = 'train_label/train_3.txt'
    val_save = 'train_label/valid_3.txt'
    label_save = ''
    xywh2x1y1w1h1_yolov3(logpath,train_save,val_save,label_save)
    '''

    #生成yolov5训练数据
    logpath = "label.txt"                     #合成数据时保存的label路径
    data_path = 'yolov5_data'                 #保存的路径名称
    img_sour = 'img'                          #生成数据保存的路径
    xywh2x1y1w1h1_yolov5(logpath,img_sour,data_path)