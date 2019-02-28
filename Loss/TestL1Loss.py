#coding=utf-8

import torch
import torch.nn as nn

#介绍文章：https://blog.csdn.net/zhangxb35/article/details/72464152
x = torch.randn((4,5))
y = torch.randn((4,5))

loss_fn = nn.L1Loss(reduction='sum')
loss = loss_fn(x,y)

print('x', x, x.size() )
print('y',y,y.size())
print('loss',loss,loss.size())

# -----------------
# BCEloss
import torch.nn.functional as F

x1 = torch.randn((4,5))
y1 = torch.randn((4,5)).random_(2)
loss_fn1 = nn.BCELoss(reduction='sum')
loss1 = loss_fn1(torch.sigmoid(x1),y1)

print('x1', x1, x1.size() )
print('y1',y1, y1.size())
print('loss1',loss1,loss1.size())

# 样本量不均衡的情况下，loss中weight可以设置权重
class_weight = torch.FloatTensor([1, 10])# 这里正例比较少，因此权重要大一些
target = torch.FloatTensor(3, 4).random_(2)
weight = class_weight[target.long()] # (3, 4)
loss_fn = torch.nn.BCELoss(weight=weight, reduce=False, size_average=False)

