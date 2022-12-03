# -*- coding: utf-8 -*-
# @File:deep_learning_test1.py
# @Author:south wind
# @Date:2022-10-09
# @IDE:PyCharm
# import xlwings as xw
# import pandas as pd
# wb=xw.Book()
# sht=wb.sheets['sheet1']
# dataset =[[1,2,4],[3,4,5]]
# df = pd.DataFrame(dataset,columns=['密度','含糖率','类别'])
# sht.range('A1').value=df
# %%
import os  # 解决 rebuild tensorflow with the compiler flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 解决 rebuild tensorflow with the compiler flags


import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model
from keras.optimizers import Adam

# from keras.losses import mean_squared_error, mean_absolute_error
# 拟合不同函数
# %%
# prepare data
X = np.linspace(0, 8, 4000)
X = np.reshape(X, [-1, 1])
test_func1 = lambda x: x + 5
test_func2 = lambda x: x ** 3 + x ** 2 + 3
test_func3 = lambda x: np.sqrt(x) + 7 + x ** 2
test_func4 = lambda x: np.power(x,4) + 6
Y = test_func1(X)
Y2 = test_func2(X)
Y3 = test_func3(X)
Y4 = test_func4(X)
# plt.figure(figsize=(4, 3))
# # 作图1
# plt.subplot(221)
# plt.plot(X, Y, '.')
# # 作图2
# plt.subplot(222)
# plt.plot(X, Y2)
# # 作图3
# plt.subplot(223)
# plt.plot(X, Y3)
# # 作图4
# plt.subplot(224)
# plt.plot(X, Y4)
# plt.show()
# %%

# simple regression network
x_in = layers.Input(shape=(1,))  # 1表示特征个数；第一层的shape(特征个数)需要给出，剩余的层则会自动判断
# 构建一个全连接层（隐藏层），"8"为神经元个数
x = layers.Dense(100, kernel_regularizer=None, bias_regularizer=None)(x_in)
x = layers.Activation('relu')(x)  # 激活函数
for i in range(1):  # 训练次数,相当于全连接层增加
    x = layers.Dense(100, kernel_regularizer=None, bias_regularizer=None)(x)  # 全连接层
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Activation('relu')(x)  # 激活函数
# 输出层
x = layers.Dense(1, kernel_regularizer=None, bias_regularizer=None)(x)  # "1"代表输出值是一维的
x_out = layers.Activation('relu')(x)  # 输出函数
reg_M = Model(x_in, x_out)  # 构建整个模型框架
reg_M.summary()  # 输出模型各层的参数情况

# %%
# train the model

# optimizer表示优化器，loss表示损失函数
# 多个输出可以有多个loss，可以用一个dict来声明{'output_a':loss_func_1, 'output_b':loss_func_2}
reg_M.compile(loss='mean_absolute_error',
              optimizer=Adam(learning_rate=1e-3))  # compile来进行编译
# bacth_size表示每次梯度更新的样本数，默认是32
# epochs模型训练的迭代数
# validation_split表示将训练集中的多少设置为测试集
# shuffle是否在每轮迭代前混洗数据
# verbose日志展示，0表示不在标准输出流输出日志信息、1显示进度条、2每个epoch输出一行记录
history = reg_M.fit(X, Y, batch_size=40, epochs=80,
                    validation_split=0.2, shuffle=True)  # fit进行训练
plt.figure()  # 新建一个图
plt.plot(history.history['loss'], label='train-loss')  # 训练集上的loss
plt.plot(history.history['val_loss'], label='val-loss')  # 测试集上的loss
plt.legend()
plt.show()
# %%
# prepare test data
X_test = np.linspace(8, 10, 10)[..., None]
Y_test = test_func1(X_test)

# check prediction
Y_pred = reg_M.predict(X_test)  # x_test为即将要预测的测试集
Y_train_pred = reg_M.predict(X)  # 模型训练好后再用训练集预测一下
plt.figure()  # 再新建一个图
plt.plot(X, Y, 'k-')
plt.plot(X, Y_train_pred,
         'r-')  # 训练集预测值 [fmt]='[color][marker][line]'r--'表示红色实线；用一个字符串来定义图的基本属性，如：颜色（color），点型（marker），线型（lifestyle）
plt.plot(X_test, Y_test, 'k--',label='test')  # 测试集真实值
plt.plot(X_test, Y_pred, 'r--',label='prediction')  # 测试集预测值
plt.legend()  # 添加图例
plt.axvline(x=8)  # 表示在绘图轴上添加垂直线
# x:x=8数据坐标中的x位置以放置垂直线
# ymin:y轴上的垂直线起始位置，它将取0到1之间的值，0是轴的底部，1是轴的顶部
# ymax:y轴上的垂直线结束位置，它将取0到1之间的值，0是轴的底部，1是轴的顶部
# **kwargs:其他可选参数可更改线的属性，例如
# 改变颜色，线宽等
plt.xlim([0, 10])  # 获取或设置当前轴的x-limits，即x的显示范围
plt.show()
# %%
