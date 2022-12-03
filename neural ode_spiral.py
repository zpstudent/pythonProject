# -*- coding: utf-8 -*-
# @File:neural ode_spiral.py
# @Author:south wind
# @Date:2022-11-28
# @IDE:PyCharm
import tensorflow as tf
features=tf.constant([12,13,24,35])
label=tf.constant([1,0,1,0])
dataset=tf.data.Dataset.from_tensor_slices((features,label))#feature与label配对
print(dataset)
for element in dataset:
    print(element)
#梯度计算--tf.GradientTape()
with tf.GradientTape() as tape:
    w=tf.Variable(tf.constant([2,3,3,5],dtype=tf.float32))
    loss=tf.pow(w,2)
grad=tape.gradient(loss,w)
print(grad)
#enumerate,遍历所有元素
#tf.ont_hot 独热编码;例如对于三分类
classes=3
labels=tf.constant([1,0,2])
output=tf.one_hot(labels,depth=classes)
print(output)
#tf.nn.softmax  将输出变为概率值
y=tf.constant([1.23,3.45,6.32,7.56])
y_pro=tf.nn.softmax(y)
print(y_pro)
#assign_sub 参数更新，在此之前，要先用tf.Variable定义变量w为可训练的（可自更新的）
w=tf.Variable(tf.constant([1,2,3,4]))
w.assign_sub(tf.fill((4,),1))#等价于每个元素w=w-1
print(w)
#tf.argmax 返回某维度最大值索引号，axis=0返回每一列（经度）最大值索引，axis=1返回每一行（纬度）
#%%
from sklearn import datasets
from pandas import DataFrame
import pandas as pd
# print(datasets.load_iris())
x_data=datasets.load_iris().data
y_data=datasets.load_iris().target
x_data=DataFrame(x_data,columns=['花萼长度','花萼宽度','花瓣长度','花瓣宽度'])
#设置列名对齐
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
#显示数据v
print('x_data add index:\n',x_data)
x_data['类别']=y_data
print('x_data add a column:\n',x_data)
'''神经网络搭建步骤：
    1。准备数据：数据集读取；数据集乱序；生成训练集和测试集；配对
    2。搭建网络
    3。参数优化
    4。测试效果
    5。acc/loss可视化
'''
#%%
#tf.where(条件语句，A，B)条件语句真返回A，假返回B
#np.random.RandomState.rand(纬度) 维度为空，返回标量
import numpy as np
rdm=np.random.RandomState(seed=1)#seed=常数每次生成随机数相同
a=rdm.rand()
b=rdm.rand(2,3)
print(a,b)
#np.vstack()将两个数组按垂直方向叠加；
# np.mgrid[起始值：结束值：步长，起始值：结束值：步长，。。。]
#x.ravel()将x变为一维数组
#np.c_[数组1，数组2，。。。]使返回的间隔数值点配对
x,y,z=np.mgrid[1:3:1,2:4:0.5,3:6:1]
grid=np.c_[x.ravel(),y.ravel(),z.ravel()]
print(x.ravel())
print(x,'\n',y,'\n',z)
print(grid)
#%%
'''指数衰减学习率=初始学习率*学习率衰减率^（当前轮数/多少轮衰减一次）'''
#激活函数
# tf.nn.sigmoid,tf.nn.tanh,tf.nn.relu,tf.nn.leaky_relu解决relu函数在负数上为零的问题
'''
    建议：
    1。首选relu激活函数
    2。学习率设置较小值
    3。输入特征标准化，满足标准正态分布
    4。初始参数中心化，让随机生成的参数满足以0为均值，sqrt（2/当前层输入特征个数）为标准差的正态分布
'''
'''
损失函数：
    1。损失函数：1。均方误差loss_mse=tf.reduce_mean(tf.square(y_-y))；2。交叉熵：表征两个概率分布之间的距离，H=-sum(y*ln(y))  tf.losses.categorical_crossentropy(y,y_)
    2。softmax与交叉熵相结合，先用softmax变为符合概率分布的值，再计算交叉熵 tf.nn.softmax_cross_entrop_with_logits(y_,y)
    3。自定义损失函数
'''

'''
过拟合与欠拟合：
    欠拟合解决办法：
    1。增加输入特征项
    2。增加网络参数
    3。减少正则化参数
        引入模型复杂度指标，即给w加权值，弱化数据的噪声（一般不正则化b）
        L1正则化大概率会使很多参数变为0，因此该方法可通过稀疏参数，降低复杂度
        L2正则化会使参数接近0但不为零，因此可通过减小参数值的大小降低复杂度
    过拟合解决办法：
    1。数据清洗
    2。增大训练集
    3。采用正则化
    4。增大正则化参数
文件：p29——regularizationfree
    p29——regularizationcontain
'''

'''
神经网络参数优化器
    待优化参数w，损失函数loss，学习率lr，每次迭代一个batch，t表示当前batch迭代的总次数
    1。计算t时刻损失函数关于当前参数的梯度
    2。计算t时刻一阶动量和二阶动量，一阶动量：与梯度相关的函数；二阶动量：与梯度平方相关的函数
    3。计算t时刻下降梯度
    4。计算t+1时刻参数
    常见优化器：
    1。SGD（无动量）p32.sgd.py
    2。SGDM(增加了一阶动量） p34——sgdm.py
    3.Adagrad （在SGD基础上增加了二阶动量）p36_adagrad.py
    4.RMSProp (在SGD基础上增加了二阶动量）p38_rmsprop.py
    5.Adam (同时结合SGDM一阶动量和RMSProp二阶动量） p40.adam.py
'''

'''
使用八股搭建神经网络
    六步法：
    1。import
    2.train,test
    3.model=tf.keras.models.Sequential(逐层描述每层网络，相当于走了一边前向传播） 
    4.model.complie（配置训练方法，告知使用哪种优化器，哪个损失函数，哪个评价指标）
    5.model.fit（执行训练过程，告知训练集和测试集的输入特征和标签，告知每个batch是多少，告知要迭代多少次数据集
    6.model.summary（打印出网络结构和参数统计）

示例：
    1.class3-p8_iris_sequential.py(上层输出就是下层输入的网络结构）
    2.用自定义层 class mymodel（Model）class3-p11_iris_class.py(关于跳连的神经网络结构)
    3.MNIST手写数字识别 
        可视化数据集：p13_mnist_datasets.py
        训练：p14_mnist_sequential.py/p15_mnist_sequential.py
    4.fsahion数据集

'''
#%%
'''
数据增强：
tf.keras.preprocessing.image.ImageDataGenerator(
rescale=所有数据将乘以该数值
rotation_range=随机旋转角度数值
width_shift_range=随机宽度偏移量
height_shift_range=随机高度偏移量
水平翻转：horiaontal_filp=是否随机水平翻转
随机放缩：zoom_range=随机缩放的范围[1-n,1+n])
注意：输入数据为四维数据 

断点存取：
读取模型：model.load_weight(路径文件名）
保存模型：cp_callback=keras.callbacks.ModelCheckpoint(
    filepath=文件路径名
    save_weight_only=True/False
    save_best_only=True/False
    )
history=model.fit(...,callbacks=[cp_callback]) 

参数提取，提取可训练参数：
model.trainable_variables 返回模型中可训练的参数
设置print输出格式：
np.set_printoptions(threshold=超过多少省略显示）
 
可视化loss/acc   p23_mnist_train_ex5.py 背下来
'''

'''
给物识图（预测）p28 _mnist_app.py--class4
1。复现模型 model=keras.models.Sequential(..)
2.加载参数  model.load_weight(model_save_path)
3.预测结果  result=model.predict(x_predict)
'''

'''
用CNN实现离散数据的分类（以图像分类为例）
1。感受野
    输出特征图上每个像素点在输入特征图上的映射区域
    常用两层3*3的卷积核替换一层5*5的卷积核
2。全零填充
    padding='same'/'valid'
3。tf描述卷积层
    keras.layers.Conv2D(
    filters=卷积核个数
    kernel_size=卷积核尺寸（核高，核宽）
    strides=滑动步长（纵向步长，横向步长），默认为1
    padding='same'/'valid'是否使用全零填充
    activation='relu' or 'sigmoid' or 'tanh' or 'softmax'如果这一层卷积后还有批标准化操作，则在此处不写激活函数
    input_shape=(高，宽，通道数）输入特征图的维度
4。批标准化（对一个batch的数据进行标准化，BN）
    标准化：使数据符合以0为均值，1为标准差的分布
    批标准化：以batch为单位进行标准化
        批标准化后，第k个卷积核的输出特征图（feature map）中第i个像素点
        但批标准化会使激活函数丧失非线性特征，故会再引入两个可训练参数，缩放因子和偏移因子
        位置：卷积---批标准化---激活
        代码：keras.layers.BatchNormalization()
5.池化：用于减少特征数据量。
    最大值池化可提取图片纹理，均值池化可保留背景特征
    keras.layers.maxpool2D/AveragePooling2D
6.舍弃
    在训练时，将一部分神经元按照一定概率从神经网络中暂时舍弃。神经网络使用时，被舍弃的神经元恢复连接
    keras.layers.Dropout(舍弃的概率）
卷积神经网络：
    卷积--批标准化--激活--池化--舍弃（CBAPD）
例子：cifar10数据集--p27_cifar10_baseline.py
    (),()=tf.keras.datasets.cifar10.load_data()   
'''

'''
六个经典的卷积神经网络

'''