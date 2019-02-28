#coding=utf-8
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

def draw_hist(mylist,title='demo', x_label='x',y_label='y'):
    plt.hist(mylist,100)
    plt.xlabel(x_label)
    # plt.xlim(x_min,x_max)
    plt.ylabel(y_label)
    # plt.ylim(y_min,y_max)
    plt.show()


train_data = pd.read_csv("/Users/liudengtao/PycharmProjects/PyTorch/Linear Regression/adult.csv");
train_data.describe().T


# 可以同时group by 多个字段，如：adult.groupby(by=['native-country','sex'])
train_data['income'].replace(' <=50K',0,inplace=True)
train_data['income'].replace(' >50K',1,inplace=True)

# 直方图分析
train_data.info()
for i in train_data.columns:
    if train_data[i].dtype == np.int64 :
        max = train_data[i].values.max();
        min = train_data[i].values.min();
        draw_hist(train_data[i],i,'x_'+i,'y_'+i)
    else:
        train_data[i].value_counts().plot(kind='hist',title=i)

# feature 与 label
x_train=train_data.select_dtypes(include=['int64']).drop(['income'],axis=1)
y_train=train_data['income']


learning_rate = 0.001
epoch=100

w = torch.rand(6, dtype=torch.float64, requires_grad=True)

data_x = x_train.irow(0)
x=torch.from_numpy(data_x.values).double().view(1,6)



# print(w)

def lr_func(x):
    # print (x.dtype)
    # print(w.dtype)
    # return 1 / (1 + math.exp(-w*x))
    return  x * w.view(6,1)

def loss(x , y):
    y_pred = lr_func(x);
    print('y_pred' , y_pred)
    return (y-y_pred) * (y-y_pred)

for i in range(epoch):
    for j in range(x_train.size):
        data_x = x_train.irow(j)
        data_y = y_train.irow(j)
        l = loss(torch.from_numpy(data_x.values).double().view(1,6), data_y)
        l.backward()
        w = w - learning_rate * w.grid
        print(l, w)