# pytorch的torch.nn模块:子模块包含创建各种NN体系结构所需的构建块（module模块），在其他框架中称为layer层
# PyTorch模块都是从基类`nn.Module`继承而来的Python类。
# 模块可以具有一个或多个参数（`Parameter`）实例作为属性，这些参数就是在训练过程中需要优化的张量（在之前的线性模型中即w和b）。
# 模块还可以具有一个或多个子模块（`nn.Module`的子类）属性，并且也可以追踪其参数。
# 子模块必须是顶级属性（top-level attributes），而不能包含在list或dict实例中
import torch.nn as nn
import torch

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)    #摄氏度
t_u = torch.tensor(t_u)    #未知单位度数
t_un = 0.1 * t_u

# linear_model = nn.Linear(11,1)
# nn.Linear(in_feature,out_feature,bias)
# in_feature： int型, 在forward中输入Tensor最后一维的通道数
# out_feature： int型, 在forward中输出Tensor最后一维的通道数
# bias默认为Ture
# out = nn.Linear(linear_model)
# in_feature需要和out的output一致

# print(t_un.shape)  torch.Size([11])
# print(linear_model(t_un))
# tensor([0.7854], grad_fn=<AddBackward0>)
# Parameter containing:
# tensor([[-0.2176, -0.2729,  0.1893,  0.2085,  0.2341,  0.2698, -0.1373, -0.1607,
#           0.1244, -0.1138, -0.2203]], requires_grad=True)


# 使用一组参数调用`nn.Module`实例最终会调用带有相同参数的名为`forward`的方法，`forward`方法会执行前向传播计算；不过在调用之前和之后还会执行其他相当重要的操作。
'''
# Module.call的实现：
def __call__(self,*input,**kwargs):    #__call__():把实例对象变为可调用对象
#**kwargs:将一个可变的关键字参数的字典传给函数实参，同样参数列表长度可以为0或为其他值。
    for hook in self._forward_pre_hooks.values():
        hook(self,input)
    result = self.forward(*input,**kwargs)
    for hook in self._forward_pre_hooks.values():
        hook_result = hook(self,input,result)
        # ...
    for hook in self._backward_hooks.values():
        # ...
    return result
# 如果直接使用.forward(...)很多hook将无法正确调用
'''
# hook:知乎作者机器学习入坑者：
'''import torch
import torch.nn as nn


class TestForHook(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=2, out_features=2)
        self.linear_2 = nn.Linear(in_features=2, out_features=1)
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.initialize()

    def forward(self, x):
        linear_1 = self.linear_1(x)
        linear_2 = self.linear_2(linear_1)
        relu = self.relu(linear_2)
        relu_6 = self.relu6(relu)
        layers_in = (x, linear_1, linear_2)
        layers_out = (linear_1, linear_2, relu)
        return relu_6, layers_in, layers_out

    def initialize(self):
        """ 定义特殊的初始化，用于验证是不是获取了权重"""
        self.linear_1.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1], [1, 1]]))
        self.linear_1.bias = torch.nn.Parameter(torch.FloatTensor([1, 1]))
        self.linear_2.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1]]))
        self.linear_2.bias = torch.nn.Parameter(torch.FloatTensor([1]))
        return True

# 1：定义用于获取网络各层输入输出tensor的容器
# 并定义module_name用于记录相应的module名字
module_name = []
features_in_hook = []
features_out_hook = []


# 2：hook函数负责将获取的输入输出添加到feature列表中
# 并提供相应的module名字
def hook(module, fea_in, fea_out):
    print("hooker working")
    module_name.append(module.__class__)
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None

# 3：定义全部是1的输入
x = torch.FloatTensor([[0.1, 0.1], [0.1, 0.1]])

# 4:注册钩子可以对某些层单独进行
net = TestForHook()
net_chilren = net.children()
for child in net_chilren:
    if not isinstance(child, nn.ReLU6):
        child.register_forward_hook(hook=hook)

# 5:测试网络输出
out, features_in_forward, features_out_forward = net(x)
print("*"*5+"forward return features"+"*"*5)
print(features_in_forward)
print(features_out_forward)
print("*"*5+"forward return features"+"*"*5)


# 6:测试features_in是不是存储了输入
print("*"*5+"hook record features"+"*"*5)
print(features_in_hook)
print(features_out_hook)
print(module_name)
print("*"*5+"hook record features"+"*"*5)

# 7：测试forward返回的feautes_in是不是和hook记录的一致
print("sub result")
for forward_return, hook_record in zip(features_in_forward, features_in_hook):
    print(forward_return-hook_record[0])'''

# nn.Linear构造函数接受三个参数：输入特征(简单理解为样本中包含多少个数字)的数量，输出特征的数量，线性模型是否包含偏差（默认为True）
# 此处特征的数量是指输入和输出张量的尺寸

# print(linear_model.weight)
# tensor([[ 0.1078, -0.2073, -0.0696, -0.1037, -0.2206,  0.2539, -0.0259, -0.1203,
#          -0.2107, -0.2519,  0.0166]], requires_grad=True)
# print(linear_model.bias)
# Parameter containing:
# tensor([0.2148], requires_grad=True)

# 可以用一些输入来调用这个模块：
# x = torch.ones(11)
# linear_model(x)

# t_c = torch.tensor(t_c).unsqueeze(1)
# t_u = torch.tensor(t_u).unsqueeze(1)    #torch.Size([11, 1])

# linear_model = nn.Linear(11,1)
# optimizer = torch.optim.SGD(
#     linear_model.parameters(),
#     lr=1e-2
# )

# 之前需要自己创建参数并将其作为第一个参数传递给optim.SGD。现在可以使用parameters方法获取任何nn.Module或其子模块的参数列表
# print(list(linear_model.parameters()))
# [Parameter containing:
# tensor([[ 0.1976,  0.1702, -0.1088,  0.2584,  0.0037,  0.0535, -0.1728,  0.0028,
#          -0.1006, -0.0987,  0.1981]], requires_grad=True), Parameter containing:
# tensor([-0.2435], requires_grad=True)]
# 次调用递归调用到模块的init构造函数中定义的子模块中，并返回遇到的所有参数的列表。




n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)  # Integer类型作用求不大于number 的最大整数   val验证集
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]
# 获得可用于从数据张量构建训练集和验证集的索引：
train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u

'''linear_model1 = nn.Linear(9,1)
linear_model2 = nn.Linear(2,1)
def training_loop(n_epochs, optimizer1, optimizer2, model1,model2,loss_fn,
                  train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):

        t_p_train = model1(train_t_u)   #t_p_train:1  #train_t_u:9
        loss_train = loss_fn(t_p_train, train_t_c)   #train_t_c 9

        t_p_val = model2(val_t_u)   #val_t_u:2  t_p_val: 1
        loss_val = loss_fn(t_p_val, val_t_c)   #val_t_c: 2

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss_train.backward()
        optimizer1.step()
        optimizer2.step()

        if epoch == 1 or epoch % 1000 == 0:
            print('Epoch %d, Training loss %.4f, Validation loss %.4f' % (
                epoch, float(loss_train), float(loss_val)))
# 训练循环几乎没有改变，除了现在你不再明确地将params传递给model，因为model本身在内部保存有Parameters。

# nn常见的损失函数：nn.MSELoss均方误差（即loss_fn的定义）.


optimizer1 = torch.optim.SGD(
    linear_model1.parameters(),
    lr=1e-2)
optimizer2 = torch.optim.SGD(
    linear_model2.parameters(),
    lr=1e-2)
training_loop(
    n_epochs = 3000,
    optimizer1 = optimizer1,
    optimizer2 = optimizer2,
    model1 = linear_model1,
    model2 = linear_model2,
    loss_fn = nn.MSELoss(), # 不再使用自己定义的loss
    train_t_u = train_t_un,  #9
    val_t_u = val_t_un,   #val_t_un tensor:(2)  val_t_u
    train_t_c = train_t_c,   #9
    val_t_c = val_t_c)       #2
'''
# print()
# print(linear_model1.weight)
# print(linear_model1.bias)
# print(linear_model2.weight)
# print(linear_model2.bias)

'''Epoch 1, Training loss 216.5046, Validation loss 111.5696
Epoch 1000, Training loss nan, Validation loss 111.5696
Epoch 2000, Training loss nan, Validation loss 111.5696
Epoch 3000, Training loss nan, Validation loss 111.5696

Parameter containing:
tensor([[nan, nan, nan, nan, nan, nan, nan, nan, nan]], requires_grad=True)
Parameter containing:
tensor([nan], requires_grad=True)
Parameter containing:
tensor([[-0.5254, -0.1040]], requires_grad=True)
Parameter containing:
tensor([-0.4427], requires_grad=True)'''



# 最后一个步骤：用NN代替线性模型作为近似函数。使用神经网络不会产生更高质量的模型，因为我们这个温度校准问题背后的过程本质上是线性的。
# 重新定义模型，将所有其他内容（包括损失函数）保持不变
# 构建最简单的NN：（一个线性模块+一个激活函数）隐藏层，最后将输入传入另一个线性模块。

# nn提供了一种通过nn.Sequential容器串联模块的简单方法
# seq_model = nn.Sequential(
#     nn.Linear(1,13),
#     nn.Tanh(),
#     nn.Linear(13,1)
# )
# print(seq_model)
# Sequential(
#   (0): Linear(in_features=1, out_features=13, bias=True)
#   (1): Tanh()
#   (2): Linear(in_features=13, out_features=1, bias=True)
# )

# 得到的模型的输入是作为nn.Sequential的参数的第一个模块所指定的输入，然后将中间输出传递给后续模块，并输出最后一个模块返回的输出。
# 该模型将1个输入特征散开为13个隐藏特征，然后将通过tanh激活函数，最后将得到的13个数字线性组合为1个输出特征。
# 调用model.parameters()可以得到第一线性模块和第二线性模块中的权重和偏差
# print(param.size for param in seq_model.parameters())

# 在调用model.backward()之后，所有参数都将被计算其grad，优化器在调用optimizer.step()期间更新参数值

# 当检查由几个子模块组成的模型的参数时，用named_parameters识别参数：
# for name,param in seq_model.named_parameters():
#     print(name,param.shape)
# 0.weight torch.Size([13, 1])
# 0.bias torch.Size([13])
# 2.weight torch.Size([1, 13])
# 2.bias torch.Size([1])

# Sequential中每个模块的名称都是该模块在参数中出现的顺序；接受OrderedDict作为参数，给Sequential的每个模块明明
from collections import OrderedDict

# seq_model = nn.Sequential(OrderedDict([
#     ('hidden_linear',nn.Linear(1,8)),
#     ('hidden_activation',nn.Tanh()),
#     ('output_linear',nn.Linear(8,1))
# ]))

# print(seq_model)
# Sequential(
#   (hidden_linear): Linear(in_features=1, out_features=8, bias=True)
#   (hidden_activation): Tanh()
#   (output_linear): Linear(in_features=8, out_features=1, bias=True)
# )

# 可以通过访问子模块来访问特定参数：
# print(seq_model.output_linear.bias) Parameter containing:tensor([0.2011], requires_grad=True)

seq_model1 = nn.Sequential(OrderedDict([
    ('hidden_linear',nn.Linear(9,8)),
    ('hidden_activation',nn.Tanh()),
    ('output_linear',nn.Linear(8,1))
]))
optimizer1 = torch.optim.SGD(seq_model1.parameters(), lr=1e-3) # 为了稳定性调小了梯度

seq_model2 = nn.Sequential(OrderedDict([
    ('hidden_linear',nn.Linear(2,3)),
    ('hidden_activation',nn.Tanh()),
    ('output_linear',nn.Linear(3,1))
]))
optimizer2 = torch.optim.SGD(seq_model2.parameters(), lr=1e-3)

def training_loop(n_epochs, optimizer1,  optimizer2, model1,model2,loss_fn,
                  train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):

        t_p_train = model1(train_t_u)   #t_p_train:1  #train_t_u:9
        loss_train = loss_fn(t_p_train, train_t_c)   #train_t_c 9

        t_p_val = model2(val_t_u)   #val_t_u:2  t_p_val: 1
        loss_val = loss_fn(t_p_val, val_t_c)   #val_t_c: 2

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss_train.backward()
        optimizer1.step()
        optimizer2.step()


        if epoch == 1 or epoch % 1000 == 0:
            print('Epoch %d, Training loss %.4f, Validation loss %.4f' % (
                epoch, float(loss_train), float(loss_val)))
training_loop(
    n_epochs = 5000,
    optimizer1 = optimizer1,
    optimizer2 = optimizer2,
    model1 = seq_model1,
    model2 = seq_model2,
    loss_fn = nn.MSELoss(),
    train_t_u = train_t_un,
    val_t_u = val_t_un,
    train_t_c = train_t_c,
    val_t_c = val_t_c)

# print('output', seq_model2(val_t_u))
# print('answer', val_t_c)
# print('hidden', seq_model1.hidden_linear.weight.grad)
# print('hidden', seq_model2.hidden_linear.weight.grad)

# 由于t_p_train和t_p_val特征数不同，因此需要分别针对这两个设计model
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x9 and 1x8)

'''Epoch 1, Training loss 196.9233, Validation loss 142.2581
Epoch 1000, Training loss 92.2099, Validation loss 142.2581
Epoch 2000, Training loss 92.2099, Validation loss 142.2581
Epoch 3000, Training loss 92.2099, Validation loss 142.2581
Epoch 4000, Training loss 92.2099, Validation loss 142.2581
Epoch 5000, Training loss 92.2099, Validation loss 142.2581
output tensor([-0.6730], grad_fn=<AddBackward0>)
answer tensor([14.,  8.])
hidden tensor([[ 5.0169e-07,  4.8532e-07,  1.8792e-07,  5.8962e-07,  4.1722e-07,
          3.0774e-07,  5.2066e-07,  2.9222e-07,  7.0599e-07],
        [ 2.7830e-07,  2.6921e-07,  1.0424e-07,  3.2707e-07,  2.3143e-07,
          1.7071e-07,  2.8882e-07,  1.6210e-07,  3.9162e-07],
        [-6.0745e-08, -5.8762e-08, -2.2753e-08, -7.1391e-08, -5.0517e-08,
         -3.7261e-08, -6.3041e-08, -3.5382e-08, -8.5482e-08],
        [-4.5588e-07, -4.4100e-07, -1.7076e-07, -5.3577e-07, -3.7912e-07,
         -2.7964e-07, -4.7311e-07, -2.6554e-07, -6.4152e-07],
        [ 8.9002e-08,  8.6097e-08,  3.3338e-08,  1.0460e-07,  7.4015e-08,
          5.4594e-08,  9.2366e-08,  5.1841e-08,  1.2525e-07],
        [ 3.0727e-07,  2.9724e-07,  1.1510e-07,  3.6113e-07,  2.5553e-07,
          1.8848e-07,  3.1889e-07,  1.7898e-07,  4.3240e-07],
        [-4.2054e-07, -4.0681e-07, -1.5752e-07, -4.9424e-07, -3.4973e-07,
         -2.5796e-07, -4.3644e-07, -2.4495e-07, -5.9179e-07],
        [-5.5383e-07, -5.3575e-07, -2.0745e-07, -6.5090e-07, -4.6058e-07,
         -3.3972e-07, -5.7477e-07, -3.2259e-07, -7.7936e-07]])
hidden None'''

# 可以在整个数据上评估模型以查看与线性关系之间的差异
from matplotlib import pyplot as plt
t_range = torch.arange(20.,90.).unsqueeze(1)   #torch.arange(start,end,step=1):返回值介于[start,end]之间的一维张量，以step为步长等间隔取值
fig = plt.figure(dpi=100)
plt.xlabel('Fahrenheit')
plt.ylabel('Celsius')
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
# plt.plot(t_range.numpy(), seq_model2(10.1 * t_range).detach().numpy(), 'c-')
# 报错：RuntimeError: mat1 and mat2 shapes cannot be multiplied (70x1 and 2x3)
# plt.plot(t_u.numpy(), seq_model2(0.1 * t_u).detach().numpy(), 'kx')
plt.show()

