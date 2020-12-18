from cachetools import TTLCache
import multiprocessing
import base64
import bottle
import uuid
import threading
import face
import time 
import requests
import datetime
usr = face.FaceEngine()#global DeviceIds
import  json
DeviceIds = dict()
starttime = datetime.datetime.now()
url_update = 'http://huizhike.cn/100/api/device/update-time'
now_time = requests.get(url_update)
print(now_time.status_code) 
@bottle.route('/hello')
def hello():
    return 'hello word!'
@bottle.route('/main', method='POST')
def login():
#    DeviceId = bottle.request.POST.get('DeviceId')
    global DeviceIds 
    global starttime
    global now_time
    data =  bottle.request.body.read().decode('utf-8')
    data = json.loads(data)
    try :
      #  print((datetime.datetime.now() - starttime).seconds,'++++++')
        if ((datetime.datetime.now() - starttime).seconds) > 10:
            print('==========')
            starttime = datetime.datetime.now()
            if requests.get(url_update).text != now_time:
              print(requests.get(url_update).text,now_time)
              now_time = requests.get(url_update).text
              DeviceIds_t = {}
              face.get_device_id_dict(DeviceIds_t)
              DeviceIds = DeviceIds_t
              print('deviceid is upadt ____________________')
    except:
        print('device update is error')   
#    data = bottle.request.json
    if data['Data']['DeviceInfo']['DeviceUUID'] in DeviceIds :
#    if True:
        print(data['Data']['FaceInfo']['FaceQuality'])
#        if data['Data']['FaceInfo']['FaceQuality'] <6000:
#            return {"HTTP/1.1 200 OK\r\n""Connection: Close\r\n""Content-Type: application/json;charset=UTF-8\r\n""Content-Length: %d\r\n\r\n""%s"}

        face_info = {}
        img_camera_base64 = data['Data']['CaptureInfo']['FacePicture']
        face_info['device_id'] = data['Data']['DeviceInfo']['DeviceUUID']
        face_info['Position'] = data['Data']['FaceInfo']['FacePosition']
        face_info['image'] = ''
    #    face_info['create_time'] = data['Data']['CaptureInfo']['CaptureTime']
        face_info['create_time'] = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
        face_info['feature'] = ''
        face_info['quality'] =  data['Data']['FaceInfo']['FaceQuality']
        face_info['app_id'] = DeviceIds[data['Data']['DeviceInfo']['DeviceUUID']]
        img_camera  = base64.b64decode(img_camera_base64)
        file_name = uuid.uuid4().hex + '.jpg'
        face_info['filename'] = file_name
        with open('/tmp/'+file_name, 'wb') as jpg_file:
            jpg_file.write(img_camera)
        with open('log.txt', 'a+') as f:
                         f.write(time.strftime('%H:%M:%S',time.localtime()) + '  accept picture'+'\n')
        with open('old/'+ datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+'.jpg' , 'wb') as jpg_file:
             jpg_file.write(img_camera)
#        time.sleep(0.005)
        print(file_name)
        queue_camera_info.put(face_info)
#        if data['Data']['DeviceInfo']['DeviceUUID'] == 'uestctestooi':
#          with open('old/'+file_name , 'wb') as jpg_file:
#            jpg_file.write(img_camera)
#        print(face_info['Position'],face_info['quality'])

        return {"HTTP/1.1 200 OK\r\n""Connection: Close\r\n""Content-Type: application/json;charset=UTF-8\r\n""Content-Length: %d\r\n\r\n""%s"}
    else:
#       http://47.93.61.253:9000/1/api/device/fetch?id=3
#        url = 'http://huizhike.cn/1/api/device/fetch?id=' + data['Data']['DeviceInfo']['DeviceUUID']
##        url = 'http://47.93.61.253:9000/1/api/device/fetch?id=' + '1'
#        receive = requests.get(url)
#        if receive.text != 'null':
#            print('device fetch',receive.text)
#            face.get_device_id_dict(DeviceIds,eval(receive.text)['app_id'])
#            if data['Data']['DeviceInfo']['DeviceUUID'] in DeviceIds:
##                if data['Data']['FaceInfo']['FaceQuality']<6000:
##                     return {"HTTP/1.1 200 OK\r\n""Connection: Close\r\n""Content-Type: application/json;charset=UTF-8\r\n""Content-Length: %d\r\n\r\n""%s"}
#                face_info = {}
#                print(data['Data']['FaceInfo']['FaceQuality'])
#                img_camera_base64 = data['Data']['CaptureInfo']['FacePicture']
#                face_info['device_id'] = data['Data']['DeviceInfo']['DeviceUUID']
#                face_info['Position'] = data['Data']['FaceInfo']['FacePosition']
#                face_info['image'] = ''
#                face_info['feature'] = ''
#                face_info['create_time'] = data['Data']['CaptureInfo']['CaptureTime']
#                face_info['quality'] =  data['Data']['FaceInfo']['FaceQuality']
#                face_info['app_id'] = DeviceIds[data['Data']['DeviceInfo']['DeviceUUID']]
#                img_camera  = base64.b64decode(img_camera_base64)
#                file_name = uuid.uuid4().hex + '.jpg'
#                face_info['filename'] = file_name
#                with open('/tmp/'+file_name, 'wb') as jpg_file:
#                    jpg_file.write(img_camera)
#                queue_camera_info.put(face_info)
##                face.get_device_id_dict(DeviceIds) 
#                return {"HTTP/1.1 200 OK\r\n""Connection: Close\r\n""Content-Type: application/json;charset=UTF-8\r\n""Content-Length: %d\r\n\r\n""%s"}
#    #        Face_info.put( data['img'])
#            else:
                return {"HTTP/1.1 200 OK\r\n"}

if __name__ == "__main__":
#    executor = ThreadPoolExecutor(max_workers=4)
# 通过submit函数提交执行的函数到线程池中，submit函数立即返回，不阻塞
#    Features, names = face.load_facebank('facebank1')
    queue_aliged_face = multiprocessing.Queue()
    queue_camera_info = multiprocessing.Queue()
    queue_camera_img = multiprocessing.Queue()
    queue_face_features = multiprocessing.Queue()
    Face_info = multiprocessing.Queue()
    queue_f1 = multiprocessing.Queue()
    queue_f2 = multiprocessing.Queue()
    queue_face_result = multiprocessing.Queue()
    queue_add = multiprocessing.Queue()
    queue_reduce = multiprocessing.Queue()
    queue_update = multiprocessing.Queue()
    queue_update_feature = multiprocessing.Queue()
    queue_update_deviceids = multiprocessing.Queue()
    cache_total = {}
#    TTLCache(maxsize=10000, ttl=3600)
    delta = 'delta'
    full = 'full'
    cache = TTLCache(maxsize=10000, ttl=30)
    
    Features_dict = {}
    Features = {}
    face.get_device_id_dict(DeviceIds)    
    face.update_zsync(full) 
    face.update_zsync(delta)
    face.get_feature(full,queue_update,queue_reduce)
    face.get_feature(delta,queue_update,queue_reduce)
#    Features_dict[3] = {}
#    Features_dict[3]['face'] = Features_dict1[3]['face']
#    Features_dict[3]['face2'] = Features_dict1[3]['face2']
#    face.dic2array(Features_dict,Features)
#    face.update_feature(Features_dict,Features,queue_update,queue_reduce,queue_add,queue_update_feature)
#    face.update_feature(Features_dict,Features,queue_get_id)
#    Get_feature(queue_aliged_face,queue_face_features)
#    p2 = Process(target=Face_align, args=(queue_camera_face,queue_camera_face ,queue_face_landmark))
#    face.process(queue_camera_info,queue_camera_img)
#    face.Face_align(queue_camera_img,queue_camera_face,queue_face_landmark,queue_aliged_face)
#    usr.Get_feature(queue_aliged_face,queue_face_features)
#    face.Get_face_compare_result(usr,queue_face_features,Features,Feature_index)
#    print(names[Feature_index.get()+1])
    P_face_align = multiprocessing.Process(target=face.Face_align, args=(queue_camera_img,queue_aliged_face))
    P_Get_face_compare_result = multiprocessing.Process(target=face.Get_face_compare_result, args=(usr,queue_face_features,Features,queue_face_result,queue_update_feature))
    task4 = multiprocessing.Process(target=face.return_faceinfo, args=(queue_face_result,cache,queue_add))
    task5 = multiprocessing.Process(target=face.update_feature, args=(Features_dict,queue_update,queue_reduce,queue_add,queue_update_feature))
    task3 = multiprocessing.Process(target=face.time_task,args=(queue_update,queue_reduce,queue_update_deviceids))
#    task3 = multiprocessing.Process(target=face.time_task,args=(delta,full,Features_dict,Features,queue_get_id))
#    P_face_align.start()
#    P_Get_face_compare_result.start()
#    Get_feature(F_input,Feature)
#    task1 = multiprocessing.Process(target=face.process,args=(queue_camera_info,queue_camera_img,))
    task1 = threading.Thread(target=face.process,args=(queue_camera_info,queue_camera_img,))
    task2 = threading.Thread(target=face.Get_feature,args=(usr,queue_aliged_face,queue_face_features,))
#    task2 = multiprocessing.Process(target=face.Get_feature,args=(usr,queue_aliged_face,queue_face_features,))
#    task3 = threading.Thread(target=face.time_task,args=(delta,full,Features_dict,Features,queue_get_id))
#    task4 = threading.Thread(target=face.return_faceinfo, args=(queue_face_result,cache,queue_get_id,))
#    task5 = threading.Thread(target=face.update_feature, args=(Features_dict,Features,queue_get_id,))
    
    P_face_align.start()
    P_Get_face_compare_result.start()
    task1.start()
    task2.start()
    task3.start()
    task4.start()
    task5.start()

#    time_task(delta,full,Features_dict,Features,queue_get_id)
#    print(queue_camera_info.qsize())
#    print(queue_camera_img.qsize())
#    print(queue_aliged_face.qsize())
#    print(queue_face_features.qsize())
#    print(queue_face_result.qsize())
#    print(queue_get_id.qsize())
    
#    p3.start()
    bottle.run(host='0.0.0.0', port=8000,debug=True)
    
##    print(F.qsize())
    
#    print(Feature.qsize())
#    print(F_input.qsize())
#    print(F_input.qsize())
#    a = F_input.get()











