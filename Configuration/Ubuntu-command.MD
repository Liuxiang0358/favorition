# Ubuntu常用命令
## 1.系统文件配置
> 
| 目录            | 内容                              | 
| --------------- | ---------------------------------- |
| /bin            |构建最小系统所需要的命令，最常用的命令| 
| /boot           | 内核和启动文件                     | 
| /dev            | 各种设备文件                       |  
| /etc            | 系统软件的启动和配置文件            | 
| /home           |用户的主目录                        | 
| /lib            | C编译器的库                        | 
| /media          | 可移动介质的安装点                  |    
| /opt            | 可选的应用软件包                    | 
| /proc           |	进程的映像                         | 
| /root           | 超级用户root的主目录                | 
| /tmp            |临时文件存放点                       |  
|/usr             | 	非系统的程序和命令                | 
|/var             | 	系统专用的数据和配置文件           |
## 2.系统文件配置
>
* sudo apt-get update        ##更新所有软件包
* apt-get  install   下载并安装软件包
* apt-get  upgrade 下载并安装本系统上已有的软件包的最新版
* apt-get  remove 卸载特定的软件包
* apt-get source 下载特定软件的源代码
* apt-get clean 删除所有的已下载的包文件
* dpkg -l |grep ^rc|awk ‘{print $2}’ |tr [”"n”] [” “]|sudo xargs dpkg -P -  清除所以删除包的残余配置文件
* sudo apt-get autoclean   清除所以删除包的残余配置文件
* uname -a   查看内核
* cat /etc/issue   查看Ubuntu版本
* lsusb  查看USB设备  
* sudo ethtool eth0  查看网卡状态
* df -h   查看硬盘剩余空间
* free -m  查看当前的内存使用情况
## 3.压缩和解压文件
> 
* tar -zxvf xxx.tar.gz   解压缩 xxx.tar.gz
* tar -jxvf xxx.tar.bz2  解压缩 xxx.tar.bz2
* tar -zcvf xxx.tar.gz aaa bbb  压缩   
* tar -jcvf xxx.tar.bz2 aaa bbb  压缩
* sudo apt-get install rar unrar   安装
* sudo ln -f /usr/bin/rar /usr/bin/unrar   
* unrar x aaaa.rar  解压
