import torch
import torch.nn as nn
import torch.autograd as autograd
m = nn.Conv1d(16, 33, 3, stride=2)
input = autograd.Variable(torch.randn(20, 16, 50))
output = m(input)
print(output)