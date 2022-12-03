# -*- coding: utf-8 -*-
# @File:AlexNet.py
# @Author:south wind
# @Date:2022-12-03
# @IDE:PyCharm
import keras.datasets.mnist
import tensorflow as tf
from keras import layers,models,optimizers
x_train,y_train=keras.datasets.mnist.load_data()
model=models.Sequential([
    layers.Conv2D(filters=96,kernel_size=(3,3),strides=1,padding='valid'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(3,3),strides=2),

    layers.Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='valid'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(3,3),strides=2),

    layers.Conv2D(filters=384,kernel_size=(3,3),strides=1,padding='same'),
    layers.Activation('relu'),

    layers.Conv2D(filters=384,kernel_size=(3,3),strides=1,padding='same'),
    layers.Activation('relu'),

    layers.Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='same'),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(3,3),strides=2),

    layers.Flatten(),
    layers.Dense(2048,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2048,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10,activation='softmax')
])
model.compile(
    optimizer=optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)
model.fit(x_train,y_train, batch_size=30, epochs=10, validation_split=0.2, validation_freq=1)
model.summary()