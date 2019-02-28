#coding=utf-8
# written by yidi on 2019.2.13
import numpy as np
import matplotlib.pyplot as plt

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

def forward(x, w):
    return x*w

def loss(y_predic, y):
    return (y_predic - y) * (y_predic-y)


w_list=[]
loss_list=[]
for w in np.arange(0,4.1,0.1):
    sum_loss = 0
    for x,y in zip(x_data,y_data):
        y_predic = forward(x,w)
        loss1 = loss(y_predic,y)
        sum_loss = sum_loss+loss1

    w_list.append(w)
    loss_list.append(sum_loss/3)

for i in np.arange(0,41,1):
    print(w_list[i],loss_list[i])


plt.plot(w_list,loss_list)
plt.xlabel("w")
plt.ylabel("loss")
plt.show()