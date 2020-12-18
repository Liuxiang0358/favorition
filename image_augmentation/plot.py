# import cv2
# f = open("log_3.txt","r")   #设置文件对象
# data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
# f.close()  
# # data = data[-50000:]
# print(len(data))
# label = []
# filelist = []
# k = 323349
# for file in data[k:k+1]:
# #    try :   
#             filecopy = file
#         #    print(i)
#             file = file.replace('\n','')
#             bbox  = file.split(',')
#             bbox[4] = int(file.split(',')[2]) +  int(file.split(',')[4])
#             bbox[5] = int(file.split(',')[3]) +  int(file.split(',')[5])
#             name = 'result_3/' + filecopy.split(',')[0]
#             print(name)
#             box = (int(bbox[2]),int(bbox[4]),int(bbox[3]),int(bbox[5]))
#             print(box)
#             img = cv2.imread(name)
#             # print(img)
#             img = cv2.rectangle(img, (int(bbox[2]), int(bbox[3])), (int(bbox[4]), int(bbox[5])), (255, 255, 255), thickness=2)
#             # cv2.imshow('input_image', img)
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()
#             cv2.imwrite('a.jpg',img)
#             # print([filename,text],'\n')
#             # print('data/custom/images/'+filename,'\n')
# # cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
import os
file_list0 = os.listdir('mixdata/JPEGImages')
file_list1 = os.listdir('result_3')
delet_list = []
for  file in file_list0:
    if  file  in file_list1:
        delet_list.append(file)
print(delet_list)
for delet in delet_list:
    os.remove('result_3/'+delet)