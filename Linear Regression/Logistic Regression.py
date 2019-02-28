#coding=utf-8
import torch
import torch.nn as nn

x = torch.Tensor([[1.0,1.0],[2.0,2.0],[3.0,3.0],[4.0,4.0]]).requires_grad_(True)
y = torch.Tensor([[0.0],[0.0],[1.0],[1.0]])

learning_rate=0.001
epoch = 100

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2,1)

    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model()
loss_fn = nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(100):
    y_pred = model(x)

    l = loss_fn(y_pred, y)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    print ('Epoch [{}/{}], Loss: {:.4f}'.format(1000 , epoch, l.item()))
    print("weights", model.state_dict())
    print("grad", x.grad)