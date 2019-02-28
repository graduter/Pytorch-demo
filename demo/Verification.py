from __future__ import print_function
import torch
import numpy as np

x = torch.rand(5, 3)
print(x)

print(torch.cuda.is_available())

x = torch.empty(5,3)
print(x)

z = torch.zeros(5,3, dtype=torch.long);

t = torch.tensor([5,3])

x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1,8)

#pytorch 与 numpy的互换,当其中一个发生变化后，另外一个映射变量也会发生变化
numpy_x = x.numpy()
x.add_(1)

m = np.ones(5)
tensor_m = torch.from_numpy(m)
