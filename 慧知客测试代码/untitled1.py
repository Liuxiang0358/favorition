# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:15:28 2020

@author: 67273
"""


import requests
 
# 请求百度网页
response = requests.get("http://huizhike.cn/100/api/device/update-time")
 
#打印出服务器响应的header信息
print("打印出服务器响应的header信息:",response.headers)
#打印出服务器响应的状态码
print("打印出服务器响应的状态码:",response.status_code)
#打印出响应信息
print("打印出响应信息:",response.text)
#以json格式打印出响应信息
#print(response.json())
 
print("打印出request",response.request)
print("打印出请求的cookie：",response.cookies)

