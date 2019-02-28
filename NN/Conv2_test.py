#coding=utf-8

import torch
from torch.autograd import Variable
##单位矩阵来模拟输入
input=torch.ones(1,1,5,5)
input=Variable(input)
x=torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,groups=1)
out=x(input)
print(out)
print(list(x.parameters()))

f_p=list(x.parameters())[0]
f_p=f_p.data.numpy()
print("the result of first channel in image:", f_p[0].sum()+(0.3255))
