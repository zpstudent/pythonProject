# -*- coding: utf-8 -*-
# @File:deep_learning_test5.py
# @Author:south wind
# @Date:2022-11-01
# @IDE:PyCharm
# print('hello,world!')
'''图像标注：labelme,labelbee
    keras:image augmentation
    论文：自动预处理图像spatial transformer network
    1d：stn
    kalman filtering
'''
'''
from: https://keras.io/examples/vision/mnist_convnet/
'''
# %%

# 解决 rebuild tensorflow with the compiler flags
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras import layers
from tensorflow import keras
import numpy as np
import os  # 解决 rebuild tensorflow with the compiler flags

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# %%
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# %%
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name='Conv_1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='MP_1'),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name='Conv_2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='MP_2'),
        layers.Flatten(name='Flat'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax", name='Dense'),
    ]
)

model.summary()

# define some callback functions
cb_tfb = TensorBoard(
    log_dir="./tf_board",
    update_freq="epoch",
    histogram_freq=3,
    embeddings_freq=3,
)

checkpoint_filepath = './tmp/checkpoint'
cb_ckpt = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# %%
# training
batch_size = 256
epochs =5

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"])

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    callbacks=[cb_tfb])

# evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# %%
'''
Check out some wrong predictions
'''
pred = model.predict(x_test)
# print(pred.shape)
pred_idx = np.argmax(pred, axis=1)
# print(pred_idx.shape)
true_idx = np.argmax(y_test, axis=1)
wrong_idx = []
for i in range(10):
    wrong_idx.append(np.where(np.logical_and(pred_idx != i, true_idx == i))[0])  # 输出pred_idx中哪些是预测错误的,输出的是id
# print(wrong_idx[1].shape,wrong_idx[1])
select_id = 0
# print(x_test[wrong_idx[1][0]].shape)
# %%
fig, ax = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    row = i // 5
    col = i % 5
    ax[row][col].imshow(x_test[wrong_idx[i][select_id]])
    ax[row][col].set_axis_off()
    ax[row][col].set_title('Predicted as {}'.format(
        pred_idx[wrong_idx[i][select_id]]))
# %%
'''
Check out some intermediate features  查看模型的中间相关信息
'''

layers_to_check = ['Conv_1', 'Conv_2']
feature_model = Model(model.input, [model.get_layer(
    'Conv_1').output, model.get_layer('Conv_2').output])

sample_to_check = x_test[wrong_idx[5][select_id]][None, ...]
# print(sample_to_check.shape)
sample_features = feature_model(sample_to_check)
# print(len(sample_features),sample_features)
sample_features = [s for s in sample_features]
# print(len(sample_features),sample_features[0].shape,sample_features[1].shape)
fig, ax = plt.subplots(8, 4, figsize=(12, 12))
for i in range(32):
    row = i // 4
    col = i % 4
    ax[row][col].imshow(sample_features[0][0, ..., i], cmap='gray')
    ax[row][col].set_axis_off()
plt.suptitle('Feature map for the first convolutional layer.')
plt.show()

fig, ax = plt.subplots(8, 8, figsize=(12, 12))
for i in range(64):
    row = i // 8
    col = i % 8
    ax[row][col].imshow(sample_features[1][0, ..., i], cmap='gray')
    ax[row][col].set_axis_off()
plt.suptitle('Feature map for the second convolutional layer.')

# %%
'''
Check out the change of predictions when inputs are transformed
'''
import cv2

sample_img = x_test[4]
image_rotated = cv2.rotate(sample_img * 255, cv2.ROTATE_90_COUNTERCLOCKWISE)
pred_rotated = model.predict(image_rotated[None, ..., None] / 255)
print(image_rotated[None,...,None].shape)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(sample_img, cmap='gray')
ax[0].set_title('Prediction: {}'.format(pred_idx[4]))
ax[1].imshow(image_rotated, cmap='gray')
ax[1].set_title('Prediction: {}'.format(np.argmax(pred_rotated)))
plt.show()
'''
Q: Produce a list of increasing angles and track the model's prediction on these rotated images.
You can stack these image into a batch and predict at once.
'''

# %%
'''旋转'''
import tensorflow as tf
rotate = [x for x in range(0, 360, 40)]  # 旋转度数
fyx, ax = plt.subplots(int(np.sqrt(len(rotate))), int(np.sqrt(len(rotate))))
n=0
for x in rotate:
    i = n // int(np.sqrt(len(rotate)))
    j = n % int(np.sqrt(len(rotate)))
    h, w = sample_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, x, 1)
    rotated = cv2.warpAffine(sample_img, M, (w, h))
    tensor_rotated=tf.cast(rotated[...,None], dtype=tf.float32)
    pred_rotated = model.predict(rotated[None, ...,None])
    # print(pred_rotated.shape)
    # print(rotated.shape,type(tf.cast(rotated,dtype=tf.float32)))
    ax[i][j].imshow(tensor_rotated, cmap='gray')
    ax[i][j].set_title(f'{np.argmax(pred_rotated)}')
    n+=1
    if n == len(rotate): break
plt.show()
