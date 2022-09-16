# -*- coding: utf-8 -*-
# @File:EM算法示例.py
# @Author:south wind
# @Date:2022-09-07
# @IDE:PyCharm
import numpy as np
'''三硬币模型'''
a,b,c = input("请输入初始值：（用逗号隔开）").split(',')
a = float(a)
b = float(b)
c = float(c)
y = [1,1,0,1,0,0,1,0,1,1]
def diedai(a,b,c,y):         #迭代器
    u = []
    for j in y:
        i = (a*(b**j)*((1-b)**(1-j)))/(a*(b**j)*((1-b)**(1-j))+((1-a)*(c**j)*((1-c)**(1-j))))
        u.append(i)
    u = np.array(u).reshape((-1,1))
    a = u.sum()/float(len(y))
    y = np.array(y).reshape((1, -1))
    b = y.dot(u)/u.sum()
    c = y.dot(1-u)/(1-u).sum()
    return a,b,c
a1,b1,c1 = diedai(a,b,c,list(y))
b1 = float(b1)
c1 = float(c1)
while np.abs(np.array([a,b,c])-np.array([a1,b1,c1])).sum() >= 1e-2:
    a,b,c = a1,b1,c1
    a1,b1,c1 = diedai(a1,b1,c1,y)
    b1 = float(b1)
    c1 = float(c1)
print(a1,b1,c1)
