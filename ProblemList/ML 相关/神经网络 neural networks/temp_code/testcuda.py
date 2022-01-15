import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
ts=time.time()

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1).cuda()
y = x.pow(3)+0.1*torch.randn(x.size()).cuda()

x , y =(Variable(x),Variable(y))

# plt.scatter(x,y)
# plt.scatter(x.data,y.data)
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden).cuda()
        self.hidden2 = nn.Linear(n_hidden,n_hidden).cuda()
        self.hidden3 = nn.Linear(n_hidden,n_hidden).cuda()
        self.hidden4 = nn.Linear(n_hidden,n_hidden).cuda()
        self.predict = nn.Linear(n_hidden,n_output).cuda()
    def forward(self,input):
        out = self.hidden1(input).cuda()
        out = torch.relu(out).cuda()
        out = self.hidden2(out).cuda()
        out = torch.sigmoid(out).cuda()
        out = self.hidden3(out).cuda()
        out = torch.relu(out).cuda()
        out = self.hidden4(out).cuda()
        out = torch.sigmoid(out).cuda()
        out =self.predict(out).cuda()

        return out

net = Net(1,20,1).cuda()
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)
loss_func = torch.nn.MSELoss().cuda()

plt.ion()
plt.show()

for t in range(5000):
    prediction = net(x).cuda()
    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%5 ==0:
        xtemp=x.cpu()
        ytemp=y.cpu()
        pretemp=prediction.cpu()
        plt.cla()
        plt.scatter(xtemp.data.numpy(), ytemp.data.numpy())
        plt.plot(xtemp.data.numpy(), pretemp.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 1, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        plt.text(0.5, 0, 'Time = %.4f' % (time.time()-ts), fontdict={'size': 20, 'color': 'red'})
        plt.text(0.5, -1, 't = %d' % t, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.05)

plt.ioff()
plt.show()