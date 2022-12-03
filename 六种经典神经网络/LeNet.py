# -*- coding: utf-8 -*-
# @File:LeNet.py
# @Author:south wind
# @Date:2022-12-03
# @IDE:PyCharm
import tensorflow as tf
from keras import layers,models,optimizers
import keras
x_train,y_train=0,0
model=models.Sequential([
    layers.Conv2D(filters=6,kernel_size=(5,5),activation='sigmoid'),
    layers.MaxPooling2D(pool_size=(2,2),strides=2),
    layers.Conv2D(filters=16,kernel_size=(5,5),activation='sigmoid'),
    layers.MaxPooling2D(pool_size=(2,2),strides=2),
    layers.Flatten(),
    layers.Dense(120,activation='sigmoid'),
    layers.Dense(84,activation='sigmoid'),
    layers.Dense(10,activation='softmax')
])
model.compile(
    optimizer=optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)
model.fit(x_train,y_train, batch_size=30, epochs=10, validation_split=0.2, validation_freq=1)
model.summary()