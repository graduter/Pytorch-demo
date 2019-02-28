#coding=utf-8
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

train_data = pd.read_csv("/Users/liudengtao/PycharmProjects/PyTorch/Linear Regression/adult.csv")


# 可以同时group by 多个字段，如：adult.groupby(by=['native-country','sex'])
train_data['income'].replace(' <=50K',0,inplace=True)
train_data['income'].replace(' >50K',1,inplace=True)

# feature 与 label
x_train=train_data.select_dtypes(include=['int64']).drop(['income'],axis=1)
y_train=train_data['income']

x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
tensor_x = torch.from_numpy(x_train.values).type(torch.FloatTensor).requires_grad_(True)
target = torch.from_numpy(y_train.values).type(torch.FloatTensor)

epoch = 10000


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(6,1)

    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        # y_pred = self.linear(x)
        return y_pred

model = Model()

criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)

for i in range(epoch):

    output = model(tensor_x)
    # print(tensor_x, output, target)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print ('Epoch [{}/{}], Loss: {:.4f} '.format(epoch , i, loss.item()))


print("weights", model.state_dict())
print("grad", tensor_x.grad)