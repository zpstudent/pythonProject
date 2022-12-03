# -*- coding: utf-8 -*-
# @File:deep_learning_test2.py
# @Author:south wind
# @Date:2022-10-14
# @IDE:PyCharm
#%%
import os  # 解决 rebuild tensorflow with the compiler flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 解决 rebuild tensorflow with the compiler flags

import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
# from keras.losses import mean_squared_error, mean_absolute_error

#%%
# prepare data
X = np.linspace(0, 0.8, 80)
X = np.reshape(X, (-1,1))
test_func = lambda x: x**2 +2
Y = test_func(X)
plt.figure()
plt.plot(X, Y)
plt.show()

#%%
# define a custom layer
# https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class PolyN(layers.Layer):
    def __init__(self, order, **kwargs):
        super(PolyN, self).__init__(**kwargs)#继承layers.layer中所有定义
        self.order = order

    def build(self, input_shape):#在知道输入维数后才创建权重
        self.coef = self.add_weight(
            shape=(self.order+1,),
            initializer="random_normal",
            trainable=True,
        )

    def call(self,inputs):#在第一次调用类时运行build，后面就不用build函数了
        out = self.coef[self.order]*inputs + self.coef[self.order-1]
        if self.order > 1:
            for i in range(self.order-2, -1, -1):
                out = inputs*out + self.coef[i]
        return out

#%%

# simple regression network
x_in = layers.Input(shape=(1,))
x_out = PolyN(2, name='poly_coef')(x_in)

reg_M = Model(x_in, x_out)
reg_M.summary()


#%%
# train the model
reg_M.compile(loss='mean_absolute_error',
              optimizer=Adam(learning_rate=1e-2))
history = reg_M.fit(X, Y, batch_size = 40, epochs = 80,
                    validation_split=0.2, shuffle = True)
plt.figure()
plt.plot(history.history['loss'], label='train-loss')
plt.plot(history.history['val_loss'], label='val-loss')
plt.legend()
plt.show()
#%%
# prepare test data
X_test = np.linspace(0.8, 1.0, 10)[...,None]
Y_test = test_func(X_test)

# check prediction
Y_pred = reg_M.predict(X_test)
Y_train_pred = reg_M.predict(X)
plt.figure()
plt.plot(X_test, Y_test,'k--',label='test')
plt.plot(X, Y, 'k-')
plt.plot(X_test, Y_pred, 'r--',label='prediction')
plt.plot(X, Y_train_pred, 'r-')
plt.legend()
plt.axvline(0.8)
plt.xlim([0,1])
plt.show()
# %%
reg_M.get_layer('poly_coef').weights
# %%

    
