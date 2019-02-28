# coding=utf-8
# written by yidi on 2019.2.14
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0],requires_grad=True)
ecoph = 100
learn_rate=0.01

def forward(x):
    return w*x

def loss(x,y):
    y_predic = forward(x)
    return (y_predic-y)*(y_predic-y)

for i in range(ecoph):
    ll_sum = 0
    for x, y in zip(x_data, y_data):
        ll = loss(x,y)
        ll.backward()
        print('w.grad=', w.grad)
        w.data = w.data - learn_rate*w.grad.data
        w.grad.data.zero_()
        ll_sum = ll_sum + ll
    print(w, ll)
