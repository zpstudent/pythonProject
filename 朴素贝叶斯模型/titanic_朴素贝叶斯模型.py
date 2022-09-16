import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
'''读取数据'''
titanic = pd.read_csv('/Users/huangjunxian/PycharmProjects/pythonProject/朴素贝叶斯模型/titanic.csv')
'''数据预处理'''
# 将sex列转化为数值特征
titanic.loc[titanic.Sex == 'male','Sex'] =0
titanic.loc[titanic.Sex == 'female','Sex'] =1
# 将embarked列转化为数值特征,并填补缺失值为3
titanic.loc[titanic.Embarked == 'S','Embarked'] = 0
titanic.loc[titanic.Embarked == 'C','Embarked'] = 1
titanic.loc[titanic.Embarked == 'Q','Embarked'] = 2
titanic.Embarked = titanic.Embarked.fillna(3)
# 将cabin列转化为数值特征
titanic.loc[~titanic.Cabin.isnull(),'Cabin'] = 1
titanic.loc[titanic.Cabin.isnull(),'Cabin'] = 0
# 使用中位数填补fare和age列
titanic.Fare = titanic.Fare.fillna(titanic.Fare.median())
titanic.Age = titanic.Age.fillna(titanic.Age.median())
# 删除无用的列
useless_cols = ['PassengerId','Name','Ticket']
titanic.drop(useless_cols,axis=1,inplace=True)
titanic.info()
'''特征筛选'''
# 区分幸存者和死亡者数据
survived = titanic[titanic.Survived == 1]
dead = titanic[titanic.Survived == 0]
# 设置画布大小,18英寸宽，8英寸高
plt.figure(figsize=(18,8))
sns.set()
# 绘画每个特征和标签Survived的相关性
for i,col in enumerate(titanic.columns.drop('Survived')):
    plt.subplot(2,4,i+1)
    survived[col].plot.density()
    dead[col].plot.density()
    plt.legend(['survived', 'dead'])
    plt.xlabel(f'{col}')
plt.show()
# 调整画布大小
plt.figure(figsize=(18,10))
# 所有特征转化为flooat型并计算特征间的相关性
corrDF = titanic.astype(float).corr()
# 根据相关性矩阵绘制热力图
sns.heatmap(corrDF,cmap='Reds',annot=True)
plt.show()
# 去掉这三个特征
useless_cols = ['Parch','SibSp','Age']
titanic_cleaned = titanic.drop(useless_cols,axis=1)
# 将数据集划分成训练集和测试集
titanic_train,titanic_test = train_test_split(titanic_cleaned,test_size=0.2,random_state=2)
# 区分训练集特征和标注
X_train = titanic_train.drop('Survived',axis=1)
y_train = titanic_train['Survived']
# 区分测试集特征和标注
X_test = titanic_test.drop('Survived',axis=1)
y_test = titanic_test['Survived']
# 实例化一个高斯朴素贝叶斯分类器
gnb = GaussianNB()
'''训练模型'''
gnb.fit(X_train,y_train)
# 使用模型对测试集进行预测
y_pred = gnb.predict(X_test)
# 计算模型在测试集上的准确率
test_sample_num = X_test.shape[0]
error_sample_num = (y_test != y_pred).sum()
accuracy = (1 - error_sample_num/test_sample_num)*100
# 打印结果
print('Number of mislabeled points out of a total{} points : {},performance{:05.2f}%'
      .format(test_sample_num,error_sample_num,float(accuracy)))
'''对朴素贝叶斯分类器进行评估'''
# 准确率
acc = metrics.accuracy_score(y_test,y_pred)*100
print('The accuracy evaluated on testset is {:05.2f}'.format(acc))
# 查准率
pre = metrics.precision_score(y_test,y_pred)*100
print('The precision evaluated on testset is {:05.2f}'.format(pre))
# 查全率
rec = metrics.recall_score(y_test,y_pred)*100
print('The precision evaluated on testset is {:05.2f}'.format(rec))
# f1得分
f1 = metrics.f1_score(y_test,y_pred)*100
print('The precision evaluated on testset is {:05.2f}'.format(f1))
# 同时获取以上所有指标
print('Classification report for classifier %s:\n%s\n'%(gnb,metrics.classification_report(y_test,y_pred)))
# 混淆矩阵
disp = metrics.plot_confusion_matrix(gnb,X_test,y_test,cmap='Reds')
# metrics.ConfusionMatrixDisplay.from_predictions(X_test,y_test)
disp.figure_.suptitle('Confusion Matrix')
plt.show()





