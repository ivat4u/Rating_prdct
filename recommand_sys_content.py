#-*- coding=utf-8 -*-
import pandas as pd
import os
import numpy as np
from sklearn.cross_validation import train_test_split
import re
from sklearn import linear_model
from sklearn.externals import joblib
#需要load_matrix文件，保留电影类别数
from load_matrix import  Num_class
#除了训练数据，记录多余数据项目数
extra_id=2

path=os.getcwd()+'\\table_train.csv'
data_table= pd.read_csv(path, header=0, names=["genres", "userId", "movieId", "rating", "title", "timestamp"])
def getmat():
    X=data_table[['genres','userId','movieId']]
    x=np.zeros(shape=(data_table.shape[0],Num_class+extra_id))
    i=0
    for item in X.values:
        list=[]
        #item都是str类型，genres必须要求list类型，所以用正则表达式转换，这里可以用lamda表达式优化
        item[0]=re.findall(r'^(\[?)(.*?)(\]?)$', item[0])[0][1]
        item[0] = re.findall(r'\[(.*)\]', item[0])
        item[0] = re.findall(r'(\d)', item[0][0])
        j=0
        for number in item[0]:
            x[i][j]=number
            j=j+1
        #保留title和id，方便追溯数据和之后修改
        for id in range(extra_id):
            x[i][j]=item[id+1]
            id=id+1
            j=j+1
        i=i+1
    Y=data_table.rating.real
    y=np.zeros(shape=(data_table.shape[0],1))
    i=0
    for item in Y:
        y[i][0]=item
        i+=1
    #可以用x.shape y.shape看两者形状,先将两者融合，之后划分训练集测试集
    return np.concatenate((x,y), axis=1)
data_get=getmat()
#利用迭代器，依次生产每个用户的数据集
def splituser(data,userId):
    for user in usersId:
        yield data_get[np.where(data_get.T[Num_class]==user)]
#记录所有id
usersId = np.unique(data_get[:, Num_class])
#生成迭代器的对象
func=splituser(data_get,usersId)
#迭代生成模型，保存每个用户的模型
for user in usersId:
    data_train=func.__next__()
    x = data_train[:, 0:Num_class]
    y = data_train[:, -1]
    x = np.insert(x, 0, values=1, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)
    # add ones column
    model = linear_model.LinearRegression(normalize=False)
    model.fit(X_train, y_train)
    y_predict=model.predict(X_test)
    #scroe分数是对RESM误差的估计，越接近0效果越好
    print(model.score(X_test,y_test))
    # save
    joblib.dump(model, os.getcwd()+'\\model\\model_user_%d.pkl'%user,compress=3)
    # restore 再次载入时使用
    #model3 = joblib.load('clf2.pkl')




