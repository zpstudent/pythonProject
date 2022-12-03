# -*- coding: utf-8 -*-
# @File:fashion.py
# @Author:south wind
# @Date:2022-11-30
# @IDE:PyCharm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator
# from keras import datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(type(x_train), x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train,x_test=x_train/255.0,x_test/255.0
# %%
# x_train = np.vstack((x_train, x_test))
# y_train = np.concatenate((y_train, y_test), axis=None)
# x_train, x_test = x_train / 255.0, x_test / 255.0
# %%
# np.random.seed(114)
# np.random.shuffle(x_train)
# np.random.seed(114)
# np.random.shuffle(y_train)
# tf.random.set_seed(114)
# %%
model = keras.models.Sequential(
    [
        layers.Flatten(),
        layers.Dense(784, activation='relu'),
        layers.Dense(784, activation='relu'),
        layers.Dense(10, activation='softmax')

    ]
)


# %%
# class mymodel(keras.Model):
#     def __init__(self):
#         super(mymodel, self).__init__()
#         self.d1 = layers.Flatten(),
#         self.d2 = layers.Dense(784, activation='relu'),
#         self.d3 = layers.Dense(784, activation='relu'),
#         self.d4 = layers.Dense(10, activation='softmax')
#
#     def call(self, x):
#         x = self.d1(x)
#         x = self.d2(x)
#         x = self.d3(x)
#         y = self.d4(x)
#         return y
# model = mymodel()
#%%
model.compile(
    optimizer=optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)
model.fit(x_train, y_train, batch_size=30, epochs=10, validation_data=(x_test,y_test), validation_freq=1)
model.summary()
# %%
