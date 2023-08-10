# Ubuntu18.04环境下配置FRP

## 1.概述
### 本文使用阿里云的公网IP地址作为VPN服务器，使用远程SSH方法访问在公司的服务器
## 2.FRP
### FRP是一个可用于内网穿透的高性能反向代理应用，支持TCP、UDP、http、HTTPS协议
## 3.设备环境
### frp服务器：阿里云应用轻量级服务器 Ubuntu
### frp客户端：在实验室的GPU服务器

### 一、服务器端配置：
    直接在release界面(https://github.com/fatedier/frp/releases)下载适合自己电脑的程序

    直接使用命令行下载：wget https://github.com/fatedier/frp/releases/download/v0.32.1/frp_0.32.1_linux_amd64.tar.gz

### 进行解压：
      tar -zxvf frp_0.32.1_linux_amd64.tar.gz
### 进入到解压目录，对于服务器端，我们主要使用frps和配置frps.ini文件，将frps.ini文件修改为：
    [common]
    bind_port = 7000           #与客户端绑定的进行通信的端口
### 保存后启动服务端：
    # 后台启动（正常使用后推荐后台启动）
    nohup ./frps -c ./frps.ini &

### 二、客户端配置：
    直接使用命令行下载：wget https://github.com/fatedier/frp/releases/download/v0.32.1/frp_0.32.1_linux_amd64.tar.gz

### 进行解压：
    tar -zxvf frp_0.32.1_linux_amd64.tar.gz
### 进入到解压目录，对于客户端，我们主要使用frpc和配置frpc.ini文件，将frpc.ini文件修改为：
    [common]
    server_addr = *.*.*.*       # 公网服务器ip
    server_port = 7000          # 与服务端bind_port一致
    
    #公网通过ssh访问内部服务器
    [ssh]
    type = tcp               # 连接协议
    local_ip = 192.168.3.48  # 内网服务器ip(127.0.0.1也行)
    local_port = 22          # ssh默认端口号
    remote_port = 6000       # 自定义的访问内部ssh端口号
    
    # 解释一下local_port和remote_port
    # 我们ssh访问的是公网ip，remote_port是公网服务器的port，访问remote_port后会转发到local_port
### 保存后启动客户端：
    # 后台启动（正常使用后推荐后台启动）
    nohup ./frpc -c ./frpc.ini &

### 三、SSH远程连接公司服务器：
    ssh -p remote_port username@公网ip
    # 举例
    ssh -p 6000 horizon@106.15.333.9

### 四、注意点
#### 防火墙的端口一定要开放