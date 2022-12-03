# -*- coding: utf-8 -*-
# @File:InceptionNet.py
# @Author:south wind
# @Date:2022-12-03
# @IDE:PyCharm
import tensorflow as tf
from keras import layers, models, optimizers
import keras.datasets.mnist
from keras import Model

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)


# %%
class ConBNRelu(Model):
    def __init__(self, ch, kernel_size=(3, 3), strides=1, padding='same'):
        super(ConBNRelu, self).__init__()
        self.model = models.Sequential([
            layers.Conv2D(ch, kernel_size=kernel_size, strides=strides, padding=padding),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])

    def call(self, x):
        x = self.model(x)
        return x


class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConBNRelu(ch, kernel_size=(1, 1), strides=strides)
        self.c2_1 = ConBNRelu(ch, kernel_size=(1, 1), strides=strides)
        self.c2_2 = ConBNRelu(ch, kernel_size=(3, 3), strides=1)
        self.c3_1 = ConBNRelu(ch, kernel_size=(1, 1), strides=strides)
        self.c3_2 = ConBNRelu(ch, kernel_size=(5, 5), strides=1)
        self.p4_1 = layers.MaxPooling2D(3, strides=1, padding='same')
        self.c4_2 = ConBNRelu(ch, kernel_size=(1, 1), strides=strides)

    def call(self, x):
        x1 = self.c1(x)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        return x


class Inception10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(Inception10, self).__init__(**kwargs)
        self.in_channels=init_ch
        self.out_channels=init_ch
        self.num_blocks=num_blocks
        self.init_ch=init_ch
        self.c1=ConBNRelu(init_ch)
        self.blocks=keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id==0:
                    block=InceptionBlk(self.out_channels,strides=2)
                else:
                    block=InceptionBlk(self.out_channels,strides=1)
                self.blocks.add(block)
            self.out_channels *=2

        self.p1=layers.GlobalAveragePooling2D()
        self.f1=layers.Dense(num_classes,activation='softmax')

    def call(self,x):
        x=self.c1(x)
        x=self.blocks(x)
        x=self.p1(x)
        y=self.f1(x)
        return y

model=Inception10(num_blocks=2,num_classes=10)
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)
model.fit(x_train,y_train, batch_size=30, epochs=10, validation_data=(x_test,y_test), validation_freq=1)
model.summary()