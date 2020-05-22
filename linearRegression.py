import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# Loss函数
def Loss(data_set, a, b):
    sum = 0
    n = data_set.size
    for x in data_set:
        sum += (a * x[0] + b - x[1])**2
    return sum / n

def linear_regression(data_set):
    m = data_set.size
    xset = []
    yset = []
    zset = []
    #  步长，0.2是多次调试后得出的最优步长
    learn_rate = 0.2
    # 初始参数设置为0
    a = 0.0
    b = 0.0
    # 训练次数
    epochs = 1000
    for epoch in range(epochs):
        sum_a = 0
        sum_b = 0
        # 求偏导数 dL/da, dL/db
        for x in data_set:
            sum_a += x[0] * (x[0] * a - x[1] + b)
            sum_b += x[0] * a - x[1] + b
        dL_da = sum_a / m
        dL_db = sum_b / m
        
        # 梯度下降
        a -= learn_rate * dL_da
        b -= learn_rate * dL_db
        print("epoch: %d : a= %.10f, b=%.10f, Loss=%.10f" % (epoch, a, b, Loss(data_set, a, b)))
    return a, b
    
if __name__ == '__main__':

    # 数据集[x,y]
    data_set = np.array([
        [1, 3],
        [1.5, 2.7],
        [2, 4.5],
        [2.5, 4.8],
        [3, 5.4],
        [3.5, 7.2],
        [4, 7.8],
        [4.5, 8.7],
        [5, 9.7]
    ])
    
    # 调用回归方法
    a, b= linear_regression(data_set)
    print("h(x) = %.10fx + %.10f" % (a, b))
    print("Loss：%.15f" % Loss(data_set, a, b))


    # matplotlib 绘图
    x = np.arange(1, 6)
    y = a * x + b
    p_x = []
    p_y = []
    for data in data_set:
        p_x.append(data[0])
        p_y.append(data[1])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y)
    plt.scatter(p_x, p_y)
    
    plt.show()