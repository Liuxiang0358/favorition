# box里保存的是ROI感兴趣区域的坐标（x，y的最大值和最小值）
# 返回值为ROI中心点相对于图片大小的比例坐标，和ROI的w、h相对于图片大小的比例
import os
classes = ['Gas_tank']
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)
 
 
# 对于单个xml的处理
def convert_annotation(image_add):
    # image_add进来的是带地址的.jpg
    #image_add = os.path.split(image_add,' ')[1]  # 截取文件名
    image_name = image_add.split()[0]
    print(image_name)
    image_name = image_name.replace('.jpg', '')  # 删除后缀，现在只有文件名
    in_file = open('gas/xml/' + image_name + '.xml')  # 图片对应的xml地址
    out_file = open('yolov3txt/%s.txt' % (image_name), 'w')
 
    tree = ET.parse(in_file)
    root = tree.getroot()
 
    size = root.find('size')
 
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    # 在一个XML中每个Object的迭代
    for obj in root.iter('object'):
        # iter()方法可以递归遍历元素/树的所有子元素
        # difficult = obj.find('difficult').text
        # cls = obj.find('name').text
        # # 如果训练标签中的品种不在程序预定品种，或者difficult = 1，跳过此object
        # if cls not in classes or int(difficult) == 1:
        #     continue
        # cls_id = classes.index(cls)#这里取索引，避免类别名是中文，之后运行yolo时要在cfg将索引与具体类别配对
        cls_id = 0
        xmlbox = obj.find('bndbox')
 
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
 
if not os.path.exists('yolov3txt'):#不存在文件夹
    os.makedirs('yolov3txt')
 
image_adds = os.listdir('./gas/jpg/')
print(image_adds)
for image_add in image_adds:
    image_add = image_add.strip()
    
    convert_annotation(image_add)
 
print("Finished")
