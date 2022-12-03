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
X = np.linspace(0, 8, 80)
X = np.reshape(X, [-1, 1])
test_func = lambda x: x ** 2 + 5
Y = test_func(X)
plt.figure()
plt.plot(X, Y, '.')
plt.show()
# %%

# simple regression network
x_in = layers.Input(shape=(1,))
x = layers.Dense(8, kernel_regularizer=None, bias_regularizer=None)(x_in)
x = layers.Activation('tanh')(x)
for i in range(1):  # 训练次数
    x = layers.Dense(8, kernel_regularizer=None, bias_regularizer=None)(x)  # 全连接层
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Activation('tanh')(x)  # 激活函数
x = layers.Dense(1, kernel_regularizer=None, bias_regularizer=None)(x)
x_out = layers.Activation('relu')(x)

reg_M = Model(x_in, x_out)
reg_M.summary()

# %%
# train the model
reg_M.compile(loss='mean_absolute_error',
              optimizer=Adam(learning_rate=1e-3))
history = reg_M.fit(X, Y, batch_size=40, epochs=80,
                    validation_split=0.2, shuffle=True)
plt.figure()
plt.plot(history.history['loss'], label='train-loss')
plt.plot(history.history['val_loss'], label='val-loss')
plt.legend()

# %%
# prepare test data
X_test = np.linspace(0.8, 1.0, 10)[..., None]
Y_test = test_func(X_test)

# check prediction
Y_pred = reg_M.predict(X_test)
Y_train_pred = reg_M.predict(X)
plt.figure()
plt.plot(X, Y, 'k-')
plt.plot(X, Y_train_pred, 'r-')
plt.plot(X_test, Y_test, 'k--')
plt.plot(X_test, Y_pred, 'r--')
plt.axvline(0.8)
plt.xlim([0, 1])
# plt.show()
# %%
