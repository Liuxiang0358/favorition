# ubuntu18.04配置notebook远程连接
## 1.生成配置文件
> jupyter notebook --generate-config
## 2.生成连接密钥
1.  使用password指令生成密钥，生成的密钥在一个文件里
> jupyter notebook password
2. 第二种方法：在python环境中生成密钥
- ipython   
- from notebook.auth import passwd
- passwd()
> 会生成一组密钥 “&&&&&&&&&&&”
> 复制下来
## 3.修改配置文件
  vim ~/.jupyter/jupyter_notebook_config.py 
* 按以下规则修改 
> c.NotebookApp.ip='*'　　　　# 允许所有ip访问  
  c.NotebookApp.password = u'sha:ce...' # 第2步复制的密钥(注意第一个单引号前加小写u) 
  c.NotebookApp.open_browser = False　　# 可改可不改 默认打开浏览器 
  c.NotebookApp.port =8888 　　　　　　　#外部连接的端口  
  c.NotebookApp.allow_remote_access = True　　# 不改就报错，被坑了一下午  
