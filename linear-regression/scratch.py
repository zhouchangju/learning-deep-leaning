"""
线性回归的实现
通过该代码，了解整个流程，以及几个关键的超参数对结果的影响，便于我们在训练模型时能够更加有效地调参。
参考：chapter_linear-networks/linear-regression-scratch.ipynb
"""
import random
import torch

"""
生成数据集
y=Xw+b+噪声
"""
def synthetic_data(w, b, num_examples):  
    """生成服从正态分布的随机数，均值为0，标准差为1，形状是根据样本数和权重来确定的"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    """通过矩阵相乘得到y的值"""
    y = torch.matmul(X, w) + b
    """加入噪声"""
    y += torch.normal(0, 0.01, y.shape)
    """将一维的y数组转化为一个len(y)的列向量"""
    return X, y.reshape((-1, 1))


"""
随机生成feature和label的批次数据

在线性回归中，Feature（特征）是指用于预测目标变量的输入变量，也称为自变量。
而Label（标签）是指我们要预测的目标变量，也称为因变量。
在训练模型时，我们使用一组已知的特征和对应的标签来训练模型，然后使用训练好的模型来预测新的标签值，以便我们能够对未知数据进行预测。
例如，我们可以使用房屋的面积、卧室数量、浴室数量等特征来预测房屋的价格，其中价格就是我们的标签。
"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break


"""
线性回归模型
"""
def linreg(X, w, b):  
    return torch.matmul(X, w) + b

"""
均方损失
"""
def squared_loss(y_hat, y):  
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

"""
小批量随机梯度下降

SGD 的全称是随机梯度下降（Stochastic Gradient Descent），是一种常用的优化算法，用于训练机器学习模型。
它通过不断地沿着梯度的反方向更新模型参数，从而最小化损失函数并使模型更好地拟合训练数据。
与传统的梯度下降算法相比，随机梯度下降在每次迭代中只使用一小部分训练数据，因此更加高效。
"""
def sgd(params, lr, batch_size):  
    """注意关掉梯度，因为梯度是累加的，如果不关掉，会导致梯度爆炸"""
    with torch.no_grad():
        for param in params:
            """沿着梯度的反方向更新模型参数(反向传播算法)，因此是减法"""
            param -= lr * param.grad / batch_size
            """更新完参数后，将梯度清零，如果不清零，梯度会累积，导致参数更新出现错误"""
            param.grad.zero_()

"""真实参数"""
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# print('features:', features[0],'\nlabel:', labels[0])

"""初始化权重和偏置，注意设置梯度为True"""
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
raw_w = w.clone()
raw_b = b.clone()

"""
学习率
学习率设置太小：计算梯度的操作会很多，代价昂贵
学习率设置太大：震荡，甚至无法收敛
"""
lr = 0.03
num_epochs = 3
batch_size = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    """1、获取小批量数据"""
    for X, y in data_iter(batch_size, features, labels):
        """2、计算模型输出"""
        y_hat = net(X, w, b)
        """3、计算损失"""
        print('before loss: w', w, ', w.grad=', w.grad, 'b', b,', b.grad=', b.grad)
        l = loss(y_hat, y)
        print('after loss: w', w, ', w.grad=', w.grad, 'b', b,', b.grad=', b.grad)
        """4、计算梯度"""
        l.sum().backward()
        print('after backward: w', w, ', w.grad=', w.grad, 'b', b,', b.grad=', b.grad)
        """5、更新参数"""
        sgd([w, b], lr, batch_size)
        print('after sgd: w', w, ', w.grad=', w.grad, 'b', b,', b.grad=', b.grad)
    with torch.no_grad():
        """6、计算训练集的损失"""
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'初始w={raw_w}，最终w={w}，生成数据的w={true_w}；初始b={raw_b}，最终b={b}，生成数据的b={true_b}')

"""比较真实参数和通过训练学到的参数来评估训练的成功程度"""
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')