# -*- coding: utf-8 -*-
# @File:后向算法示例(P213-1).py
# @Author:south wind
# @Date:2022-09-08
# @IDE:PyCharm
import numpy as np

'''初始化数据'''
A = np.array([0.5, 0.2, 0.3, 0.3, 0.5, 0.2, 0.2, 0.3, 0.5]).reshape((3, 3))
B = np.array([0.5, 0.5, 0.4, 0.6, 0.7, 0.3]).reshape(3, -1)
pai = np.array([0.2, 0.4, 0.4])
T = 4  # 长度
O = [1, 0, 1, 0]  # 观测序列，1代表红，0代表白
beita = np.zeros((T, 3))  # 后向概率矩阵
beita[T - 1] = 1  # 规定最后时刻的所有状态的后向概率均为1
'''计算后向概率，并储存为在后向概率矩阵里'''
for t in range(T - 2, -1, -1):
    for i in range(3):
        for j in range(T - 1):
            if O[t + 1] == 0:
                beita[t, i] += A[i, j] * B[j, 1] * beita[t + 1, j]
            else:
                beita[t, i] += A[i, j] * B[j, 0] * beita[t + 1, j]
P = 0
for i in range(3):
    if O[0] == 1:
        P += pai[i] * B[i, 0] * beita[0, i]
    else:
        P += pai[i] * B[i, 1] * beita[0, i]
print(f'后向概率矩阵为：\n{beita}')
print(f'观测序列概率为：{P}')
