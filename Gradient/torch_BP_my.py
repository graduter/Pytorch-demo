# coding=utf-8
# written by yidi on 2019.2.14
import torch

x_data = torch.tensor([1.0, 2.0, 3.0],requires_grad=True)
y_data = torch.tensor([2.0, 4.0, 6.0],requires_grad=True)

w = torch.tensor([1.0, 1.0, 1.0],requires_grad=True)
ecoph = 100
learn_rate=0.0001

def forward(x):
    return w*x

def loss(x,y):
    y_predic = forward(x)
    return (y_predic-y)*(y_predic-y)

def gradient(x, y):
    l = loss(x,y)
    l.backward(learn_rate*l)
    return x.grad

for i in range(ecoph):
    ll = loss(x_data,y_data)
    delt_w = gradient(x_data, y_data)
    w = w - delt_w
    print(w, ll, delt_w)
