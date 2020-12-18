# Ubuntu18.04环境下配置深度学习环境
## 1.安装ssh
      sudo apt-get install openssh-server
## 2.安装NVIDIA TITAN Xp显卡驱动
### 默认安装的显卡驱动不是英伟达的驱动，所以先把旧得驱动删除掉。
      sudo apt-get purge nvidia*
### 添加Graphic Drivers PPA
      sudo add-apt-repository ppa:graphics-drivers/ppa
      sudo apt-get update
### 查看合适的驱动版本：
      ubuntu-drivers devices
### 图中可以看出推荐的是最新的430版本的驱动，安装该驱动：
      sudo apt-get install nvidia-driver-430
### 安装完毕后重启机器：
      sudo reboot
### 重启完毕运行：
      nvidia-smi
## 3.安装依赖库
      sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-devlibgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
## 4.GCC降低版本
> CUDA9.0要求GCC版本是5.x或者6.x，其他版本不可以，需要自己进行配置，通过以下命令才对GCC版本进行修改.
* 版本安装
>         sudo apt-get install gcc-5              
>         sudo apt-get install g++-5          
* 通过命令替换掉之前的版本
>      sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50
>      sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50
* 通过命令自由更换gcc、g++
>     sudo update-alternatives --config gcc
## 5.安装Anaconda和tensorflow、keras和pytorch
* 下载Anaconda
> 安装
>     bash Anaconda3-2019.03-Linux-x86_64.sh
* 更改pip源
>     mkdir ~/.pip
>     vim ~/.pip/pip.conf
>     [global]    index-url = https://pypi.tuna.tsinghua.edu.cn/simple 
* 在Anaconda中安装Python 的虚拟环境
>     conda create --name tf python=3.7 #创建tf环境
虚拟环境主要命令
>     source activate tf #激活tf环境
>     source deactivate tf #退出tf环境
>     conda remove --name tf --all #删除tf环境（全部删除）
* 在Anaconda中安装TensorFlow GPU 1.9
>     conda install tensorflow-gpu==1.9
