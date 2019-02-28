#coding=utf-8
# written by yidi on 2019.2.13
import numpy as np
import matplotlib.pyplot as plt

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

#初始化，网络参数、迭代次数、学习率
w = 1.0
epoch=100
learn_rate=0.002

def forward(x):
    return x*w

def loss(x, y):
    y_predic = forward(x)
    return (y_predic - y) * (y_predic-y)

def gradient(x,y):
    return 2*w*(x*w - y)

w_list=[]
loss_list=[]

for i in range(epoch):
    print(w)
    sum_loss = 0
    for x,y in zip(x_data,y_data):
        loss1 = loss(x,y)
        sum_loss = sum_loss+loss1
        delt_w = gradient(x,y)
        w = w - learn_rate * delt_w
    w_list.append(w)
    loss_list.append(sum_loss/3)

    print(w, sum_loss/3)

plt.plot(w_list,loss_list)
plt.xlabel("w")
plt.ylabel("loss")
plt.show()