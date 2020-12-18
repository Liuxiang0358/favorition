* 本项目进行煤气罐和电梯照片的合成

* Requirements
    python == 3.7 
    python-opencv
* 使用说明

- 打开 爬虫.py 
        将  keyword = '液化石油气罐' , 代码会进行网络爬虫获取相关照片 ，图像将会保存在 'image/' 文件夹中
- 安装 labelme标注工具，https://github.com/wkentaro/labelme#windows
        使用labelme标注工具对相应图片进行分割，标注，每一个煤气罐都需要单独进行标注 
- 打开 labelme2img.py 
        修改 input_fold 为label(json)所在的文件夹
        修改 save_fold  为保存煤气罐图片的文件夹

- 运行 图像粘贴.py 
        path_Elevator = 'nobody_deal'       ##  电梯图片路径     
        path_gas = 'img'                    ## 需要合成的图片路径
        path_result_save = 'test'           ##  保存合成图片的路径
        log_save = 'log_test.txt'           ##  保存标签的文件名
        获得合成照片，获得 .txt标注文件
- 运行 1.py 将标签转换为 yolov3 的训练格式
        logpath = "log_3.txt"    #合成照片所生成的标签文件
        train_save = 'train_label/train_3.txt'   #训练集 
        val_save = 'train_label/valid_3.txt'   #测试集
        label_save = ''   # 训练的lanel 保存路径 
