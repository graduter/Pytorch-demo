#coding=utf-8

import torch
from torch.autograd import Variable

import torch.nn as nn

input=torch.ones(1,3,7,7)
input=Variable(input)
x=torch.nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,groups=1)
out=x(input)
print(out)
print(list(x.parameters()))

f_p=list(x.parameters())[0]
f_p=f_p.data.numpy()
print(f_p[0].sum()+(-0.0113))