# -*- coding: utf-8 -*-
# @File:VGGNet.py
# @Author:south wind
# @Date:2022-12-03
# @IDE:PyCharm
import keras.datasets.mnist
from keras import layers,models,optimizers
import tensorflow as tf
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0

x_train=tf.expand_dims(x_train,axis=-1)
x_test=tf.expand_dims(x_test,axis=-1)
#%%
model=models.Sequential([
    layers.Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2),strides=2,padding='same'),
    layers.Dropout(0.2),

    layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
    layers.Dropout(0.2),

    layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(filters=256,kernel_size=(3,3),padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
    layers.Dropout(0.2),

    layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
    layers.Dropout(0.2),

    layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(512,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10,activation='softmax')
])
#%%
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)
model.fit(x_train,y_train, batch_size=30, epochs=10, validation_data=(x_test,y_test), validation_freq=1)
model.summary()