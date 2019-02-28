#coding=utf-8
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class myNet(nn.module):
    def __init__(self):
        super(myNet, self).__init__()

        self.conv1 = nn.Conv2d(1,6,(5,5))
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        # self.conv3 = nn.Conv2d(16,120,6)  #也可以作为卷积层处理
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view()