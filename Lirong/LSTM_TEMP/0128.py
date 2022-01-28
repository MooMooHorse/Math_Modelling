import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

# lstm单元输入和输出维度都是3
lstm = nn.LSTM(3, 3).cuda()
# 生成一个长度为5，每一个元素为1*3的序列作为输入，这里的数字3对应于上句中第一个3
inputs = [autograd.Variable(torch.randn((1, 3)).cuda()) for _ in range(5)]
'''
# 设置隐藏层维度，初始化隐藏层的数据
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
autograd.Variable(torch.randn((1, 1, 3)))) 
 '''

inputs = torch.cat(inputs).view(len(inputs), 1, -1).cuda()
hidden = (autograd.Variable(torch.randn(1, 1, 3).cuda()), autograd.Variable(torch.randn((1, 1, 3)).cuda())) # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)