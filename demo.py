# -*- coding: utf-8 -*-
# @File:demo.py
# @Author:south wind
# @Date:2022-09-08
# @IDE:PyCharm
# -*- coding: utf-8 -*-
# @File:前向后向概率计算示例(P213-2).py
# @Author:south wind
# @Date:2022-09-08
# @IDE:PyCharm
import numpy as np

'''初始化数据'''
A = np.array([0.5, 0.2, 0.3, 0.3, 0.5, 0.2, 0.2, 0.3, 0.5]).reshape((3, 3))
B = np.array([0.5, 0.5, 0.4, 0.6, 0.7, 0.3]).reshape(3, -1)
pai = np.array([0.2, 0.4, 0.4])
O = [1, 0, 1]  # 观测序列，1代表红，0代表白
T = len(O)  # 长度
beita = np.zeros((T, 3))  # 后向概率矩阵
alpha = np.zeros((T, 3))  # 前向概率矩阵
Gamma = np.zeros((T, 3))  # 时刻-状态矩阵
beita[T - 1] = 1  # 规定最后时刻的所有状态的后向概率均为1
'''计算后向概率，并储存为在后向概率矩阵里'''
for i in range(3):
    for t in range(T - 2, -1, -1):
        for j in range(3):
            if O[t + 1] == 0:
                beita[t, i] += A[i, j] * B[j, 1] * beita[t + 1, j]
            else:
                beita[t, i] += A[i, j] * B[j, 0] * beita[t + 1, j]
'''计算前向概率，并储存在前向概率矩阵中'''
for i in range(3):
    if O[0] == 1:
        alpha[0, i] = pai[i] * B[i, 0]
    else:
        alpha[0, i] = pai[i] * B[i, 1]
for t in range(T - 1):
    for i in range(3):
        print(alpha[t],A[:,i])
        h = sum(alpha[t] * A[:, i])
        if O[t + 1] == 1:
            alpha[t + 1, i] = h * B[i, 0]
        else:
            alpha[t + 1, i] = h * B[i, 1]
'''计算各时刻各状态的概率（行坐标表示时刻，列坐标表示状态）'''
for t in range(T):
    for i in range(3):
        Gamma[t, i] = (alpha[t, i] * beita[t, i]) / sum(alpha[t] * beita[t])
'''观测序列概率'''
P = 0
m = alpha[T - 1].sum()
for i in range(3):
    if O[0] == 1:
        P += pai[i] * B[i, 0] * beita[0, i]
    else:
        P += pai[i] * B[i, 1] * beita[0, i]

print(f'后向概率矩阵为：\n{beita}')
print(f'前向概率矩阵为：\n{alpha}')
print(f'时刻-状态矩阵为：\n{Gamma}')
print(f'观测序列概率为{P, m}')
