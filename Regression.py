import torch
import torch.nn.functional as AF
import matplotlib.pyplot as plt
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)             # 设置隐藏层  feature->hidden
        self.output = torch.nn.Linear(n_hidden, n_output)               # 设置输出层，hidden->output

    # 神经网络前馈
    def forward(self, x):
        x = AF.tanh(self.hidden(x))       # 通过隐藏层，tanh函数激活
        x = self.output(x)                # 通过输出层，回归问题输出层不用激活函数激活
        return x
    

# 初始化训练数据
x = torch.unsqueeze(torch.linspace(-1, 1, 200), dim=1)
x_np = x.data.numpy()
y = x.pow(2) + 0.2 * torch.rand(x.size()) 
y_np = y.data.numpy()


# 初始化神经网络为 1 输入-> 10隐藏-> 1输出
net = Net(1, 10, 1)
# 初始化优化器，使用SGD梯度下降方法
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
# 初始化代价函数，使用均方差 MSELoss
loss_func = torch.nn.MSELoss()


# 训练过程
for epoch in range(20000):
    prediction = net(x) # 将x通过神经网络，获得预测值
    loss = loss_func(y, prediction)  # 预测值与正确值带入代价函数计算误差
    optimizer.zero_grad()          # 梯度归零，因为上次训练的梯度会保存在optimizer中，每次必须清零梯度

    loss.backward()   #
    optimizer.step() # 梯度下降


    
    # 实时画图
    plt.cla()
    plt.scatter(x_np, y_np)
    plt.plot(x_np, prediction.data.numpy())
    plt.text(0.5, 0, "MSE LOSS: %.5f" % loss)
    plt.text(0.5, 0.1, "epoch: %d" % epoch)
    plt.pause(0.0001)
    print("Epoch: %d,      MSELoss: %.5f" % (epoch, loss))
#plt.scatter(x_np, y_np)
#plt.plot(x_np, prediction.data.numpy())
plt.show()