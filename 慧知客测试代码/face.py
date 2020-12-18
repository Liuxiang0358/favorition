#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# coding:utf-8
import  datetime
from easydict import EasyDict as edict
import utils.face_model as face_model
#from retinaface import RetinaFace
from cachetools import TTLCache
import cv2
import schedule
import requests
import random
import numpy as np 
import face_preprocess
import csv
import os
import time
import gzip
import shutil
import base64
import oss2
import json
DetectMode = {
	"ASF_DETECT_MODE_IMAGE": 0,
	"ASF_DETECT_MODE_VIDEO": 1
}

#gpuid = 0
#thresh = 0.8   
#detector = RetinaFace('./model/mnet.25', 0, gpuid, 'net3')
####################################################################################################
'''
error = {'0001':'the input image file error','0002':'No face detect'}

'''
url_aliyun = 'http://huizhike.cn'
url_signature = 'http://oss-cn-beijing.aliyuncs.com'

def un_gz(file_name):
    """解压gz包"""
    f_name = file_name.replace(".gz", "")
    #获取文件的名称，去掉
    g_file = gzip.GzipFile(file_name)
    #创建gzip对象
    with open(f_name, "wb+") as f:
        f.write(g_file.read())
    #gzip对象用read()打开后，写入open()建立的文件里。
    g_file.close()
    

########################
#获得人脸去重时间
def get_facelog():
    url = url_aliyun + '/1/api/config/facelog'
    face_log = requests.get(url)
#    np.save(os.path.join(os.getcwd(),'document/devices.npy'),face_log)
    return eval(face_log.text)['merge_minutes']

#########################
#def set_cache_ttl(cache_total):
#    store_ids = get_store_ID()
#    cache_total = manage_features(cache_total,store_ids)
#    ttl_time = get_facelog()
#    for store_id in store_ids:
#        cache_total[store_id] = TTLCache(maxsize=10000, ttl=ttl_time[store_id*60])
########################
# 
def get_store_ID():
#    url_aliyun = 
#    url = 'http://47.93.61.253:9000/0/api/app/all-id'
#    print('get_id')
    with open('log1.txt', 'a+') as f:
                f.write(time.strftime('%H:%M:%S',time.localtime())+'\n')
    url = url_aliyun + '/0/api/app/all-id'
    face_log = requests.get(url)
    return eval(face_log.text)

###################
# 更新设备ID          
# 
def get_device_id_dict(DeviceIds, store_ids = -1):
   # print('get_device_id_dict')
    if store_ids == -1:
        Store_ids = get_store_ID()
        with open('log.txt', 'a+') as f:
                f.write(time.strftime('%H:%M:%S',time.localtime())+'     get_device_id_dict'+'\n')
        for store in Store_ids:
            try:   
                url = url_aliyun +'/'+ str(store) + '/api/device/all-id'
                face_log = requests.get(url)
                if face_log.status_code == 200:
                   uuid = json.loads(face_log.text)
                   if None in uuid:
                       uuid.remove(None)
                   dic  = dict.fromkeys(uuid, store)
                   b = set(dic.keys())
                   a = set(eval(face_log.text))
                   for de in list(b - a):
                       del  dic[de]
                   DeviceIds.update(dic)
            except:
                print('error')
    else:
        store = store_ids
        url = url_aliyun +'/'+ str(store) + '/api/device/all-id'
        face_log = requests.get(url)
        if face_log.status_code == 200:
           uuid = json.loads(face_log.text)
           if None in uuid:
                uuid.remove(None)
           dic  = dict.fromkeys(uuid, store)
           DeviceIds.update(dic)

#########################
#  同步当前目录
def manage(Store_ids):
    local =  [int(i) for i in Store_ids]
    romot = [int(i) for i in os.listdir('store')]
    tmp =list(set(local) & set(romot))
    pwd = os.getcwd()
    for store_id in list(set(romot) - set(tmp)):
        shutil.rmtree(os.path.join(pwd,'store',str(store_id)))
    for store_id in list(set(local) - set(tmp)):
        os.makedirs(os.path.join(pwd,'store',str(store_id),'face/full/'))
#        os.makedirs(os.path.join(pwd,'store',str(store_id),'face2/full/'))
        os.makedirs(os.path.join(pwd,'store',str(store_id),'face/delta/'))
#        os.makedirs(os.path.join(pwd,'store',str(store_id),'face2/delta/'))

########################
## 获得远程签名
def get_signature_string(file_name,method):
    
    auth = oss2.Auth('LTAI4FsjZYVAFWCNbejMocJ6','8OsM29gYLTIh9xUti55f5JnpdNjJmN')
# Endpoint以杭州为例，其它Region请按实际情况填写。
    bucket = oss2.Bucket(auth, 'http://oss-cn-beijing.aliyuncs.com','tdx-data')
    
    # 设置此签名URL在60秒内有效。
#    print(bucket.sign_url('GET', '<yourObjectName>', 60))
    return bucket.sign_url(method,file_name, 60)
    
#    signature_string = Signature[Signature.find('?'):]
#    '?OSSAccessKeyId==<LTAI4FsjZYVAFWCNbejMocJ6>&Expires=1572767620&Signature='
#    return signature_string
#    url = 'http://tdx-data.oss-cn-beijing.aliyuncs.com/upload/1/face/4dde187a-2bf9-49bd-a549-9bf4be9278e4.jpg?OSSAccessKeyId=<LTAI4FsjZYVAFWCNbejMocJ6>&Expires=<unix time>&Signature='+signature_string

########################
## 检测zsync是否存在
def exist_zsync(dir_path):
    Your_Dir= dir_path
    Files = os.listdir(Your_Dir)
    for k in range(len(Files)):
      # 提取文件夹内所有文件的后缀
      Files[k]=os.path.splitext(Files[k])[1]
      
    # 你想要找的文件的后缀
    Str='.csv'
    if Str in Files:
      return True
    else:
      return False

########################
## 字典和索引做成数组    ***
def dic2array(Features_dict):
   
    store_ids = get_store_ID()
#    model = ['full','delta']
    with open('log.txt', 'a+') as f:
                f.write(time.strftime('%H:%M:%S',time.localtime())+'     dic2array'+'\n')
    copy = {}
#    Features_copy = manage_features(Features_copy,store_ids)
    manage_features(copy,store_ids)
    faces = ['face']

    try:
        for d in Features_dict:
            for face in faces:
                values = list(Features_dict[d][face].values())
                if len(values) > 0:
                    feature = [v[1] for v in values]
                    quality = [v[0] for v in values]
    #            Features[d][face] = np.array(list(Features_dict[d][face].keys())).astype(np.longlong),np.array(list(Features_dict[d][face].values()).astype(np.float32))
                    copy[d][face] = (np.array(list(Features_dict[d][face].keys())).astype(np.long),np.array(quality[:]),np.array(feature[:]))

        return  copy.copy()
    except:
    #    print('dic2array is error')
         pass

########################
## 管理特征   ****
def manage_features(Features,store_ids):
    a = set(store_ids)
    b = set(Features.keys())
    c = a & b
    for store_id in list(a - c):
        Features[store_id] = {}
        Features[store_id]['face'] = {}
#        Features[store_id]['face2'] = {}
    for store_id in list(b - c):
        del Features[store_id]
#    return Features.copy()
        
########################
## 从文件中读取特征
def get_feature(model,queue_update,queue_reduce):
    try:

        store_ids = get_store_ID()
        with open('log.txt', 'a+') as f:
                f.write(time.strftime('%H:%M:%S',time.localtime()) +'   get_feature'+'\n')
    #    model = ['full','delta']
    #    print('update get_feature')
    #    Features_dict = manage_features(Features_dict,store_ids)
        Features_dict = {}
        manage_features(Features_dict,store_ids)
        faces = ['face']
        Features_copy = {}
        for store_id in store_ids:
            Features_copy[store_id] = {}
            f_full_and_delta = {}
            for fa in faces:
                ids = []
                combination = []
                pwd = os.getcwd()
                file =os.path.join(pwd ,'store',str(store_id) ,fa, model,model +'.csv')
    #            file = '/home/lab507c/downlowm/Face_project/store/3/face/full/full.csv'
    #            file =os.path.join(pwd ,'store',str(store_id) ,face , model,time.strftime('%Y-%m-%d',time.localtime(time.time())) +'.csv')
                if os.path.exists(file):
                    with open(file, 'rt') as f:        # 采用b的方式处理可以省去很多问题
                        reader = csv.reader(f)
    #                    print(file)
                        for row in reader:
                            
                            try:
                                if row[1] != '':
                                    ids.append(np.long(row[0]))
    #                                print(row[1]) 
                                    quality = np.array(row[1],dtype = np.int)
                                    features =np.fromstring(base64.b64decode(row[4]),dtype = np.float32)
                                    combination.append((quality,features))  
                                else:
                                    queue_reduce.put([store_id,fa,np.long(row[0])])
                            except:
                                pass
    #                            print(row ,'error')
    #                        features.append(eval(row[1]))
    #                        features.append(row[1])
                f_full_and_delta.update(dict(zip(ids,combination)))
                Features_copy[store_id][fa] = f_full_and_delta
                Features_dict[store_id][fa].update(Features_copy[store_id][fa])
        queue_update.put(Features_dict)
    except:
#        print('get feature is error')
     pass
#    Features_copy = dic2array(Features_copy.copy)
#    return Features_dict.copy()

#def update_deviceids(queue_update_deviceids):
#    DeviceIds = {}
#    get_device_id_dict(DeviceIds)  
#    queue_update_deviceids.put(DeviceIds)
def time_task(queue_update,queue_reduce,queue_update_deviceids):
#    schedule.every(1).day.at('23:00').do(update_deviceids,queue_update_deviceids)
    schedule.every(60).minutes.do(update_zsync,'delta')
    schedule.every(120).minutes.do(update_zsync,'full')
    schedule.every(120).minutes.do(get_feature,'full',queue_update,queue_reduce)
    schedule.every(60).minutes.do(get_feature,'delta',queue_update,queue_reduce)
    while True:
       schedule.run_pending()

########################
## 
def getDate(url_face):
    
    r = requests.get(url_face + 'latest.txt') 
    with open('latest.txt', "wb") as code:
             code.write(r.content)
    with open('latest.txt', "r") as code:
          date =  code.read()   
#    os.remove('latest.txt')
    return date
       
def getYesterday(): 
    yesterday = datetime.date.today() + datetime.timedelta(-1)
    return yesterday.strftime('%Y-%m-%d')
########################
def update_zsync(model):
    print('update_zsync')
    try:
        store_ids = get_store_ID()
        with open('log.txt', 'a+') as f:
                f.write(time.strftime('%H:%M:%S',time.localtime()) + '     update_zsync'+'\n')
        manage(store_ids)
#        faces = ['face']
        i = 1
        for store_id in store_ids:
            
    #        url_face = url + '/store/'+str(store_id) + '/face/'+ model+'/'+time.strftime('%Y-%m-%d',time.localtime(time.time())) +'.csv.gz' + signature
            fa = 'face'
    #            for fa in faces:
            if model == 'full':
                url_face  = 'http://tdx-data.oss-cn-beijing.aliyuncs.com/' + 'store/'+str(store_id) + '/'+fa+'/'+ model+'/'
                date = getDate(url_face)
                url_face  =  url_face + date + '.gz'
            #    print(model,store_id )
#                print(url_face)
            else:
                url_face  = 'http://tdx-data.oss-cn-beijing.aliyuncs.com/' + 'store/'+str(store_id) + '/'+fa+'/'+ model+'/'+ datetime.date.today().strftime('%Y-%m-%d') + '.csv.gz'
             #   print(model,store_id )
#                print(url_face)
    #                url_face = get_signature_string('store/'+str(store_id) + '/'+face+'/'+ model+'/'+ datetime.date.today().strftime('%Y-%m-%d') + '.csv.gz','GET')
    #                url_face_head = get_signature_string('store/'+str(store_id) + '/'+face+'/'+ model+'/'+ datetime.date.today().strftime('%Y-%m-%d') + '.csv.gz','HEAD')
    #                url_face_zsync = get_signature_string('store/'+str(store_id) + '/'+face+'/'+ model+'/'+ datetime.date.today().strftime('%Y-%m-%d') + '.csv.zsync','GET')
    #            else:
    #                 url_face = 'http://tdx-data.oss-cn-beijing.aliyuncs.com/' + 'store/'+str(store_id) + '/'+face+'/'+ model+'/'+ datetime.date.today().strftime('%Y-%m-%d') + '.csv.gz'
    #                url_face_head = get_signature_string('store/'+str(store_id) + '/'+face+'/'+ model+'/'+ datetime.date.today().strftime('%Y-%m-%d') + '.csv.gz','HEAD')
    #                url_face_zsync = get_signature_string('store/'+str(store_id) + '/'+face+'/'+ model+'/'+ datetime.date.today().strftime('%Y-%m-%d') + '.csv.zsync','HEAD')
    #       url_face = 'http://tdx-data.oss-cn-beijing.aliyuncs.com/store/1/face/delta/2019-10-25.csv.gz'
    #            url_face2 =  get_signature_string('store/'+str(store_id) + '/face/'+ model+'/'+time.strftime('%Y-%m-%d',time.localtime(time.time())) +'.csv.gz')
            r = requests.head(url_face)
            data_length = r.headers['Content-Length']
            pwd = os.getcwd()
            os.chdir(os.path.join(pwd,'store/',str(store_id),fa,model))
            if eval(data_length) > 32000:
#                if exist_zsync(os.path.join(pwd,'store/',str(store_id),fa,model)):
#    #                    print(' > 32000' ,1)
#                    os.system('zsync -i '+ model +'.csv'+' '+ url_face.replace('gz','zsync'))
#                    try:    
#                        os.rename(time.strftime('%Y-%m-%d',time.localtime(time.time()))+'.csv', model + '.csv')
#                    except:
#                        pass
#                else:
#    #               print(' > 32000',2)
                    os.system('zsync' +' '+ url_face.replace('gz','zsync') + ' -i ' + model +'.csv' + ' -o ' +model + '.csv')
#                    try:
#                        os.rename(time.strftime('%Y-%m-%d',time.localtime(time.time()))+'.csv', model + '.csv')
#                    except:
#                         pass
    
            else:
    #           print(' < 32000')
                r = requests.get(url_face) 
                with open(model+'.csv.gz', "wb") as code:
                         code.write(r.content)
                try:
                    un_gz(model+'.csv.gz')
                except:
                    pass
                try:
                    os.remove(model+'.csv.gz')
                except:
                    pass
    #            os.rename( filename.replace('gz',''), model + '.csv')
            os.chdir(pwd)
    #        print(model +  str(i)+'/'+str(len(store_ids))+'   is update')
            i = i + 1
    except:
    #        print('zsync is error')    
            pass

## zsync http://tdx-data.oss-cn-beijing.aliyuncs.com/store/14/face2/full/2019-11-19.csv.zsync?OSSAccessKeyId=LTAI4FsjZYVAFWCNbejMocJ6&Expires=1574133345&Signature=4LtnR0mdl67xp82zvpdIysT7h9Y%3D
#def Time_tasks(Features_dict,Features,queue_get_id):
#    
#    schedule.every(2).minutes.do(update_zsync,('delta'))
#    schedule.every(2).minutes.do(update_zsync,('full'))
##    schedule.every(1).minutes.do(get_feature,(Features_dict,'delta',queue_get_id))
##    schedule.every(1).minutes.do(get_feature,(Features_dict,'full',queue_get_id))
#    schedule.every(1).minutes.do(dic2array,(Features_dict,Features))
#    
#    while True:
#        schedule.run_pending()

# struct data 
DetectOrient = {
	"ASF_OP_0_ONLY": 1,
	"ASF_OP_90_ONLY": 2,
	"ASF_OP_270_ONLY": 3,
	"ASF_OP_180_ONLY": 4,
	"ASF_OP_0_HIGHER_EXT": 5
}
engineConfigTemplate = {
	"detectFaceMaxNum": 1,
	"detectFaceOrientPriority": DetectOrient["ASF_OP_0_HIGHER_EXT"],
	"detectFaceScaleVal": 12,
	"detectMode": DetectMode["ASF_DETECT_MODE_IMAGE"],

	"functionConfig":{
		"age": True,
		"gender": True,
		"faceDetect": True,
		"faceRecognition": False,
		"liveness": True,
		"rgbLivenessThreshold": 0.8,
	}
}

engineConfig = engineConfigTemplate
def get_args(engineConfig):
    '''
    funtion:
        Setting model parameters
    parameters:
        filePath: engineConfig
    return:
        imageInfo: dict
    '''
    args = edict()
    args.imagesize=  "112,112"     ####  检测到人脸图片后的图像大小
    args.FaceMaxNum =  engineConfig["detectFaceMaxNum"]
    args.detectMode = engineConfig['detectMode']
    args.age = engineConfig['functionConfig']["age"]
    args.gender = engineConfig['functionConfig']["gender"]
    args.faceRecognition = engineConfig['functionConfig']["faceRecognition"]
    args.liveness = engineConfig['functionConfig']["liveness"]
    args.rgbLivenessThreshold = engineConfig['functionConfig']["rgbLivenessThreshold"]
    args.gpu = 0  ##########  指定GPU
    args.det = 0
    args.flip = 0
    args.filename =['jpg','bmp','png','jpeg','tif']
    return args
####################################################################################################
# image methods
def Face_align(queue_camera_img,queue_aliged_face):
    while True: 
        camera_info = queue_camera_img.get()
        try:
#            print('Face_align')
#            t = time.time()
            with open('log.txt', 'a+') as f:
                f.write(time.strftime('%H:%M:%S',time.localtime()) + 'face algn  '+ str(queue_camera_img.qsize())+'\n')
            img = camera_info['image'] 
            faces = camera_info['camera_face_img'] 
            landmarks = camera_info['landmarks'] 
            nimg = face_preprocess.preprocess(img , faces[0][:-1], landmarks.reshape((5,2)), image_size='112,112')
            del camera_info['camera_face_img'] 
            del camera_info['landmarks'] 
            del camera_info['image'] 
            camera_info['camera_align_face'] = nimg
#            cv2.imwrite('old/'+camera_info['filename'],nimg)
            queue_aliged_face.put(camera_info)
#            print('Face_align',time.time() - t)
        except:
            print('Face_align is error')
#            pass
#            time.sleep(0.1)
def process(queue_camera_info,queue_camera_img):
    from retinaface import RetinaFace
    import cv2
    import numpy as np
    gpuid = 1
    thresh = 0.7   
    detector = RetinaFace('./model/mnet.25', 0, gpuid, 'net3')
    while True:
        camera_info_process = queue_camera_info.get()
        try:
#            print('process is waiting')
#            t = time.time()
            with open('log.txt', 'a+') as f:
                f.write(time.strftime('%H:%M:%S',time.localtime()) + 'face detect ' +str(queue_camera_info.qsize())+'\n')
            img = cv2.imread('/tmp/'+camera_info_process['filename'])
            print(img.shape)
            im_shape = img.shape
            scales = [128, 128]
            target_size = scales[0]
            max_size = scales[1]
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            scales = [im_scale]
            faces, landmarks = detector.detect(img, thresh, scales=scales)
            print(landmarks.shape)
            print(faces.shape)
#            if faces.shape[0] == 1:
            camera_info_process['image'] = img
            camera_info_process['camera_face_img'] = faces
            camera_info_process['landmarks'] = landmarks
#            else:
#              print(landmarks.shape)
#              camera_info_process['image'] = img
#:              print(landmarks[0])
#              camera_info_process['camera_face_img'] = faces[0].reshape([1,5])
#              camera_info_process['landmarks'] = landmarks[0].reshape([1,5,2])
            queue_camera_img.put(camera_info_process)
#            print('process',time.time() - t)
        except:

#            pass
            print('process is error')


def Get_feature(usr, queue_aliged_face,queue_face_features):
    while True:  
        camera_info = queue_aliged_face.get()
        try:
#            t = time.time()
            with open('log.txt', 'a+') as f:
                        f.write(time.strftime('%H:%M:%S',time.localtime()) +'Get feature  '+ str(queue_aliged_face.qsize())+'\n') 
#            info_list.append(camera_info)
            face_info = {}
            face_image = np.expand_dims(np.transpose(camera_info['camera_align_face'],(2,0,1)), axis=0)
#            if queue_aliged_face.qsize() > 0:
#                for n in range(min(queue_aliged_face.qsize(),5)):
#                    camera_info = queue_aliged_face.get()
    #                    info_list.append(camera_info)
#                    face_image1 = np.expand_dims(np.transpose(camera_info['camera_align_face'],(2,0,1)), axis=0)
    #                    face_image  = F_input.get()
#                    face_image = np.concatenate((face_image,face_image1),axis=0)
            face_info['face'] = face_image
            face_feature = usr.extractFaceFeature(face_info)
            faceProperties = usr.processFaceProperty(face_info)
    #                face_feature = self.extractFaceFeature(face_info)
            n = int(len(face_feature)/512)
            feature = face_feature.reshape( n,512)
            for i in range(n):
    #                queue_face_features.put(info_list[i]['feature'] = feature[i,:])
                camera_info['feature'] = feature[i,:]
                g = faceProperties[i,0:2]
                camera_info['gender'] = np.argmax(g)
                a = faceProperties[i,2:202].reshape( (100,2) )
                a = np.argmax(a, axis=1)
                camera_info['age'] = int(sum(a))
    #                del camera_info['camera_align_face']
                queue_face_features.put(camera_info)
#            print('Get_feature',time.time() - t)
        except:
            print('Get_feature is error')    #
#            time.sleep(0.1)
#            pass
#               

            
def de_weight(face_info,cache):
    appid = face_info['app_id']
    people_id = face_info['id']

#    print(people_id, type(people_id))
#    if type(people_id) 
#    print(people_id)
    if np.long(people_id) not in cache:
        cache[people_id] = appid
        return True
    else:
        return False

def upload_oss_img(upload_file, filename):
    
    auth = oss2.Auth('LTAI4FsjZYVAFWCNbejMocJ6','8OsM29gYLTIh9xUti55f5JnpdNjJmN')
# Endpoint以杭州为例，其它Region请按实际情况填写。
    bucket = oss2.Bucket(auth, 'http://oss-cn-beijing.aliyuncs.com','tdx-data')
    # 必须以二进制的方式打开文件，因为需要知道文件包含的字节数。
    bucket.put_object_from_file(upload_file, filename)
##########################3
    
def update_feature(Features_dict,queue_update,queue_reduce,queue_add,queue_update_feature):

    while True:
        try:
            
                Flag = False
                if queue_add.qsize() > 0:
#                    for n in range(queue_add.qsize()):
                    for n in range(queue_add.qsize()):
                        face_info = queue_add.get()
                        app_id = face_info[0]
                        fa_model = face_info[1]
                        people_id = face_info[2]
                        quality = np.array(face_info[3],dtype=np.int)
                        feature = face_info[4]
                        Features_dict[app_id][fa_model][people_id] = (quality,feature)
                    Flag = True
#                    print('add face feature queue_add')
                if queue_update.qsize() > 0:
                     store_ids = get_store_ID()
                     with open('log.txt', 'a+') as f:
                         f.write(time.strftime('%H:%M:%S',time.localtime()) +'update_feature'+'\n')
                     manage_features(Features_dict,store_ids)
                     for n in range(queue_update.qsize()):
                         d_f = queue_update.get()
                     
                         for key in list(d_f.keys()):
                             if d_f[key]['face'] != {} :
                                 Features_dict[key]['face'].update(d_f[key]['face'])
                     Flag = True
#                     print('time task face feature queue_update')
                     if  queue_reduce.qsize() > 0:
                         for n in range(queue_reduce.qsize()):
                            face_info = queue_reduce.get()
                            app_id = face_info[0]
                            fa_model = face_info[1]
                            people_id = face_info[2]
                            if np.long(people_id) in Features_dict[app_id][fa_model].keys():
                                del Features_dict[app_id][fa_model][np.long(people_id)]
                         Flag = True
#                         print('time task face feature queue_reduce')
#                    

                if Flag == True:

                    Features = dic2array(Features_dict)
                    queue_update_feature.put(Features)
                   # print(Features_dict.keys())
                   # print('111',len(Features_dict['3499656975217664']['face2']))
                   # print('222',len(Features['3499656975217664']['face2'][:][0]))
                    Flag = False
        except:
            print('update_feature error')
           # pass

def return_faceinfo(queue_face_result,cache,queue_add):
    kind = {'facelog':1,'face':0}
#    i = 0
#    s = 0
    while True:
#        face_info = queue_face_result.get_nowait()
        face_info = queue_face_result.get()
#        if s==0:
#         T = time.time()
#         s = 1
#        i = i + 1
        try:
#           t = time.time()
           with open('log.txt', 'a+') as f:
                               f.write(time.strftime('%H:%M:%S',time.localtime()) +'return infortion '+ str(queue_face_result.qsize())+'\n')
           Flag = face_info['flag']
           print(face_info['create_time'])
           
           if Flag == 'facelog' :
                 files={
                'device_id':(None,face_info['device_id']),
                'face_id':(None,face_info['id']),
                'gender':face_info['gender'],
                'age':face_info['age'],
                'type':kind[Flag],
                'create_time':face_info['create_time'],
                }
                 if de_weight(face_info,cache):
    #                    headers = {'User-Agent': 'Mozilla/5.0'}
                    if face_info['quality'] > face_info['old_quality']:
                        files_put={
                        'device_id':(None,face_info['device_id']),
                        'id':(None,face_info['id']),
                        'age':(None,face_info['age']),
                        'gender':(None,face_info['gender']),
                        'feature':(None, base64.b64encode(face_info['feature'].tostring())),
                        'image':(None,'face' + '/'+face_info['filename']),
                        'quality':(None,face_info['quality']),
                        'create_time':face_info['create_time']
                        }
                        if Flag == 'facelog':
                            url = url_aliyun + '/'+ str(face_info['app_id'])+  '/api/' + 'face' + '/put'
                            r = requests.post(url ,data=files_put)
                            print(' faceput' +  'is' ,r)
                            filename = '/tmp/'+face_info['filename']
                            upload_file = 'upload/'+ str(face_info['app_id']) + '/'+'face'+'/' + face_info['filename']
                                                                #           upload_file = 'upload/'+ str(1) +'/face/' + '3.jpg'
                            upload_oss_img(upload_file, filename)

                    url = url_aliyun + '/'+ str(face_info['app_id'])+  '/api/' + 'facelog' + '/add'
        #            url = 'http://47.93.61.253:9000/1/html/face/log'
        #            headers = {'User-Agent': 'Mozilla/5.0'}
        #            url = 'http://47.93.61.253:9000/1/api/face2/add'
                    r = requests.post(url ,data=files)
#if r.status_code == 500:
#                        print(face_info['filename'])
#                         pass
                    print('upload '+ Flag + '   '+str(face_info['id']) ,'is',r)
                 else:
#                     pass
                    print('de_weight')
           else:    
    #                Flag = 'face2'
                files={
                'device_id':(None,face_info['device_id']),
                'quality':(None,face_info['quality']),
                'age':(None,face_info['age']),
                'gender':(None,face_info['gender']),
                'feature':(None, base64.b64encode(face_info['feature'].tostring())),
                'image':(None,Flag + '/'+face_info['filename']),
                'type':(None,kind[Flag]),
                'create_time':(None,face_info['create_time'])
                }
                url = url_aliyun + '/' + str(face_info['app_id'])+ '/api/' + Flag + '/add'
    #            url = 'http://47.93.61.253:9000/1/html/face/log'.
    #            headers = {'User-Agent': 'Mozilla/5.0'}
    #            url = 'http://47.93.61.253:9000/1/api/face2/add'
                r = requests.post(url ,data=files)
    #            print(r.text)
                Id = r.text
 #               print(eval(Id)[0],type(eval(Id)[0]))
                face_info['id']  = np.long(eval(Id)[0])
#                print(type(eval(Id)[0]))
                de_weight(face_info,cache)
                queue_add.put([face_info['app_id'],Flag,face_info['id'],face_info['quality'],face_info['feature']])
    #                filename = open('/tmp/'+face_info['filename'],'rb')
    #            filename = '3.jpg'
                filename = '/tmp/'+face_info['filename']
                upload_file = 'upload/'+ str(face_info['app_id']) + '/'+Flag+'/' + face_info['filename']
    #           upload_file = 'upload/'+ str(1) +'/face/' + '3.jpg'
                upload_oss_img(upload_file, filename)
                print('uoload face','is',r)
#           print('return_faceinfo',time.time() - t)

        except:
            print('return_faceinfo is error')
#             pass
 #       if i >= 999:
 #         print(i,(time.time() - T))
def Get_face_compare_result(usr,queue_face_features,Features,queue_face_result,queue_update_feature):
    update = False
    while True:
        camera_info = queue_face_features.get()
           
        with open('log.txt', 'a+') as f:
             f.write(time.strftime('%H:%M:%S',time.localtime()) + 'face compare '+ str(queue_face_features.qsize())+'\n')
        try:    
            t = time.time()
            if queue_update_feature.qsize()>0:
                try:
                    for n in range(queue_update_feature.qsize()):
                        Features = queue_update_feature.get()
                except:
                    pass
            feature = camera_info['feature']
#            print(feature)
            appid = camera_info['app_id']
            print(Features.keys(),appid)
    #            store_ids = get_store_ID()
    ##    model = ['full','delta']
    #            Features = manage_features(store_ids,Features)
#            print(Features[appid]['face2'][2])
            if len(Features[appid]['face']) != 0 :
                index_face = usr.compareFaceFeatures(feature,Features[appid]['face'][2])
                try:
                    if len(index_face) != 1:
                        index_face = index_face[0]
                except:
                    pass
                if index_face != -1 :
                    Flag = 'facelog'
                    camera_info['id'] = Features[appid]['face'][0][index_face]
                else:
                    Flag = 'face'
            else:
                Flag = 'face'
            if Flag == 'facelog':
                camera_info['old_quality'] = Features[appid]['face'][1][index_face]	
#                print(camera_info['id'])
            else:
                pass
            if Flag == 'face':
                update = True
#            print(Flag)
#            print(camera_info['id'])
            camera_info['flag'] =  Flag
            queue_face_result.put(camera_info)
#            print('Get_face_compare_result',time.time() - t)
#            print(Flag)
            if Flag == 'face':
                while queue_update_feature.qsize() == 0:
                    pass
        except:
#            time.sleep(0.1)
#            pass
            print('Get_face_compare_result is error')

        
#def load_facebank(path):
#    facebank_path = path
#    embeddings = np.load(os.path.join(facebank_path,'facebank_mxnet.npy'))
#    names = np.load(os.path.join(facebank_path,'names_mxnet.npy'))
#    return embeddings, names
    
class ImageInfo:
    def rgb(self,filepath):
        '''
    funtion:
        generate imageinfo from rgb file
    parameters:
        filePath: file path
    return:
        imageInfo: dict
        '''
        try :
            imageInfo_templete = {"format": "CP_PAF_BGR24","width": int,"height": int,"data": bytes()}
            imageInfo =  imageInfo_templete.copy()
            if filepath.split('.')[-1] in ['jpg','bmp','png','jpeg','tif'] and os.path.exists(filepath):
                 im = cv2.imread(filepath)
                 imageInfo['width'] = im.shape[1]
                 imageInfo['height'] = im.shape[0]
                 imageInfo['data'] = im
                 return imageInfo
            else:
                 return '0001'
        except:
            return None
####################################################################################################
# face methods
class  FaceEngine:
    def __init__(self):
        '''
        funtion:
            create and configure face engine
        parameters:
            engineConfig: dict, engine configurations
        return:
            None
        '''
        args = get_args(engineConfig)
        self.args = args
        self.threshold = 1.0
        self.Gender = {"female": 0,"male":1}
        self.imageInfoTemplate = {"format": "CP_PAF_BGR24","width": 600,"height": 800,"data": bytes()}
        self.model =  face_model.FaceModel(self.args)
        self.faceInfoTemplate = {"rect": {"left": int,"right": int,"bottom": int,"top": int},"face":bytes()}
        self.facePropertyTemplate = {"age": int,"gender": self.Gender["male"],"liveness": bool}

    def get_sorce(self,dist):
        face_define = ''
        if (dist>1.2):
            sorce = random.uniform(20, 40)
            face_define = '不相似'
        elif (dist >= 1.0):
            sorce = random.uniform(70, 75)
            face_define = '比较相似'
        elif (dist >= 0.9):
            sorce = random.uniform(75, 80)
            face_define = '比较相似'
        elif (dist >=0.8):
            sorce = random.uniform(80, 85)
            face_define = '很相似'
        elif (dist >=0.7):
            sorce = random.uniform(85, 90) 
            face_define = '很相似'
        else:
            sorce = random.uniform(90, 95) 
            face_define = '非常相似'
        return sorce,face_define


    def extractFaceFeature(self,  faceInfor):
        '''
        funtion:
            extract face feature from specified image part
        parameters:
            faceInfo: dict, face information
        return:
            faceFeature: bytes, face feature
        '''
        try:
            faceFeature = self.model.get_feature(faceInfor['face'])
            return faceFeature
        except:
            return None

    def compareFaceFeature(self, faceFeature1, faceFeature2):
        '''
        funtion:
            compare two face features
        parameters:
            faceFeature1: bytes, face feature 1
            faceFeature2: bytes, face feature 2
        return:
            similarScore: float, similar score
        '''
        try :
            if (len(faceFeature1) == 1) and (len(faceFeature2) == 1):
                compare = faceFeature1[0] - faceFeature2[0]
                x_norm=np.linalg.norm(compare)
                similarScore = self.get_sorce(x_norm)
                return similarScore
            return 
        except:
            return None
#    def Get_feature(self,queue_aliged_face,queue_face_features):
#        while True:
#            try:
#                face_info = {}
#                img = queue_aliged_face.get_nowait()
#                face_image = np.expand_dims(np.transpose(img,(2,0,1)), axis=0)
#                if queue_aliged_face.qsize() > 0:
#                    for n in range(min(queue_aliged_face.qsize(),70)):
#                        img = queue_aliged_face.get()
#                        face_image1 = np.expand_dims(np.transpose(img,(2,0,1)), axis=0)
#            #            face_image  = F_input.get()
#                        face_image = np.concatenate((face_image,face_image1),axis=0)
#                face_info['face'] = face_image
#                face_feature = self.extractFaceFeature(face_info)
##                face_feature = self.extractFaceFeature(face_info)
#                n = int(len(face_feature[0])/512)
#                feature = face_feature[0].reshape( n,512)
#                for i in range(n):
#                    queue_face_features.put(feature[i,:])
#            except:
#  #              time.sleep(0.3)
#                pass
    def compareFaceFeatures(self, faceFeature1, faceFeature2):
        '''
        funtion:
            compare two face features
        parameters:
            faceFeature1: bytes, face feature 1
            faceFeature2: bytes, face feature 2
        return:
            similarScore: float, similar score
        '''#
        try :
#            print(faceFeature1.shape)
#            print(faceFeature1)
#            print(faceFeature2.shape)
            if len(faceFeature2.shape) ==3:
                  faceFeature2 = np.squeeze(faceFeature2,axis = 1)
            compare = faceFeature1 - faceFeature2
            x_norm = np.linalg.norm(compare , ord=None, axis=1, keepdims=False)  
            min_list = x_norm.min()# 返回最大值
            
            if min_list < self.threshold:
                min_index =  np.argwhere(x_norm == min_list)# 最大值的索引
                min_index = np.squeeze( min_index)
            else:
                min_index = -1
            print('min_index:  ',min_list)
            return min_index
        except:
#            print('compare error')
             pass
    def processFaceProperty(self,  faceInfos):
        '''
        funtion:
            analyse face properties
        parameters:
            faceInfos: [dict], face information array
        return:
            faceProperties: [dict], face property array
        '''
        try:
#            faceProperties = self.facePropertyTemplate
#            gender,age = self.model.get_ga(faceInfos['face'])
#            faceProperties['gender'] = gender
#            faceProperties['age'] = age
#            return faceProperties
            faceproperties = self.facePropertyTemplate
            faceproperties = self.model.get_ga(faceInfos['face'])
            return faceproperties
        except:
            return None
    
