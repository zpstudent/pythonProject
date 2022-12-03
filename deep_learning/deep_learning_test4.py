# -*- coding: utf-8 -*-
# @File:deep_learning_test4.py
# @Author:south wind
# @Date:2022-11-01
# @IDE:PyCharm
"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""
# %%
import os  # 解决 rebuild tensorflow with the compiler flags

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 解决 rebuild tensorflow with the compiler flags
# from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import glob

"""
## Prepare the data
"""
# %%
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# %%
# 取出一个样本并显示出来
imgN = 50000
plt.imshow(x_train[imgN], cmap='gray')
plt.title(y_train[imgN])
plt.show()
# print(np.shape(x_train[imgN]))
# %%
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print(x_train.shape)
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
print(x_train.shape)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# %%
# convert class vectors to binary class matrices
# 将数据转换为onghot形式，num_classes=10，0-9用10维向量表示
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# %%
"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),  # stride表示步长，pad表示扩展
        layers.MaxPooling2D(pool_size=(4, 4)),  # 池化层，起到压缩作用，这里采用的是取最大值
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

"""
## Train the model
"""
# epochs表示迭代次数，batch_size表示一次性训练多少个数据
batch_size = 128
epochs = 9
# 模型编译，本质就是配置参数
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# %%
'识别手写图片'
# path=input('请输入你需要识别的数字图片的文件夹路径：')
filename_list = glob.glob('/Users/huangjunxian/PycharmProjects/pythonProject/deep_learning/data/*.png')  # 以列表形式返回所有符合条件的路径
def load_preprosess_image(image_file):  # 预处理图像
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)  # 解码RGB
    image = tf.cast(image, tf.float32)
    image = tf.image.rgb_to_grayscale(image)  # RGB转为灰度图
    image = tf.image.resize(image, [28, 28])
    image=1-(image/255)
    return image

test_dataset = tf.data.Dataset.from_tensor_slices(filename_list)
AUTOTUNE = tf.data.AUTOTUNE  # 根据计算机cpu的个数自动做并行计算
test_dataset = test_dataset.map(load_preprosess_image,
                                num_parallel_calls=AUTOTUNE)  # map使函数应用在load_preprosess_image中所有的图像上
batch_size = 3
test_count = len(filename_list)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(AUTOTUNE)  # 前台在训练时，后台读取数据，自动分配cpu
imgs = next(iter(test_dataset))  # 理论上取出的是一个batch个数的图片，shape=(batch_size,28,28,3)
print(imgs.shape, type(imgs))
for x in imgs:
    print(x.shape,type(x))
    plt.imshow(x, 'gray')
    plt.show()
for x in imgs:
    imgs = np.expand_dims(x, 0)
    pred = model.predict(imgs).reshape(-1).tolist()
    print(tf.reduce_max(pred),tf.argmax(pred),pred)

