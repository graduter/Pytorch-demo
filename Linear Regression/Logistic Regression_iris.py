#coding=utf-8
import torch
import torch.nn as nn
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from mytool import eval

# x = torch.Tensor([[1.0,1.0],[2.0,2.0],[3.0,3.0],[4.0,4.0]]).requires_grad_(True)
# y = torch.Tensor([[0.0],[0.0],[1.0],[1.0]])

data,target = datasets.load_iris(True)


# feature 与 label
np_x= data[0:100]
np_y= target[0:100]


x = torch.from_numpy(np_x).type(torch.FloatTensor).requires_grad_(True)
y = torch.from_numpy(np_y).type(torch.FloatTensor)

learning_rate=0.001
epoch = 100

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(4,1)

    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model()
loss_fn = nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(10):
    y_pred = model(x)

    l = loss_fn(y_pred, y)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    print ('Epoch [{}/{}], Loss: {:.4f}'.format(1000 , epoch, l.item()))
    # print("weights", model.state_dict())
    # print("grad", x.grad)

result = model(x);

# compare = torch.cat((result,y.view(100,1)),1)
# print compare

#roc curve
result_arr = result.squeeze().detach().numpy()
y_arr = y.numpy()
fpr, tpr, thresholds = metrics.roc_curve(y_arr, result_arr, pos_label=1)
# print(fpr,tpr,thresholds)
eval.roc_curve(fpr,tpr)