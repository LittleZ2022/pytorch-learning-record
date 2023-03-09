import torch
# 为由线性和非线性函数组成的复杂函数的导数编写解析表达式并不是一件很有趣的事情，也不是一件很容易的事情。
# 这个问题可以通过一个名为autograd的PyTorch模块来解决。PyTorch张量可以记住它们来自什么运算以及其起源的父张量，并且提供相对于输入的导数链。
# 你无需手动对模型求导：不管如何嵌套，只要你给出前向传播表达式，PyTorch都会自动提供该表达式相对于其输入参数的梯度。

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)    #摄氏度
t_u = torch.tensor(t_u)    #未知单位度数
# 1.定义模型和损失函数
def model(t_u,w,b):   #t_u:已有的位置单位的数值
    return w * t_u + b
def loss_fn(t_p,t_c):  #t_p：预测值，t_c：实际值
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()   #mean平均值

'''# 2.初始化参数张量
params = torch.tensor([1.0,0.0],requires_grad=True) #这个参数告诉PyTorch需要追踪在params上进行运算而产生的所有张量。
# 换句话说，任何以params为祖先的张量都可以访问从params到该张量所调用的函数链。
# 如果这些函数是可微的（大多数PyTorch张量运算都是可微的），则导数的值将自动存储在参数张量的grad属性中。
# 所有Pytorch张量都有一个初始为空的名为grad（梯度）的属性

# 需要做的：调用模型，计算损失值，对损失张量loss调用backwards：
loss = loss_fn(model(t_u,*params),t_c)
#“*”:用params参数收集传入是不定个数的参数，并将收集的到参数以元组的方式存储在params中，如果没有传入参数params就是个空元组
# print(loss)  tensor(1763.8848, grad_fn=<MeanBackward0>)
loss.backward()
# print(params.grad)  #tensor([4517.2969,   82.6000])
# require_grad设置为True以及组合任何函数。计算损失的导数，在这些张量（即计算图的叶节点）的grad属性中将这些导数值累计（accumulate），叠加。
# 每次迭代时（参数更新后）需要将梯度清零
if params.grad is not None:
    params.grad.zero_()'''

'''# 启用autograd的训练代码
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_() # 这可以在调用backward之前在循环中的任何时候完成
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
        params = (params - learning_rate * params.grad).detach().requires_grad_()
# 返回一个新的tensor，从当前计算图中分离下来的，但仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的tensor永远不计算其梯度，无gra
# 为了避免重复使用变量名，我们重构params参数更新行：p1 = (p0 * learningrate * p0.grad)。
# 这里p0是用于初始化模型的随机权重，p0.grad是通过损失函数根据p0和训练数据计算出来的。
# 第二次迭代：p2 = (p1 * lr * p1.grad)。p1的计算图会追踪到p0，这是有问题的，
# 因为(a)你需要将p0保留在内存中（直到训练完成），并且(b)在反向传播时不知道应该如何分配误差。
# 应该通过调用.detatch()将新的params张量从与其更新表达式关联的计算图中分离出来。
# 这样，params就会丢失关于生成它的相关运算的记忆。然后，你可以调用.requires_grad_()，这是一个就地（in place）操作（注意下标“_”），以重新启用张量的自动求导。
# 现在，你可以释放旧版本params所占用的内存，并且只需通过当前权重进行反向传播。
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params
'''
# t_un = 0.1 * t_u
# training_loop(
#     n_epochs = 5000,
#     learning_rate = 1e-2,
#     params = torch.tensor([1.0, 0.0], requires_grad=True),
#     t_u = t_un,
#     t_c = t_c)
'''
Epoch 500, Loss 7.860115
Epoch 1000, Loss 3.828538
Epoch 1500, Loss 3.092191
Epoch 2000, Loss 2.957698
Epoch 2500, Loss 2.933134
Epoch 3000, Loss 2.928648
Epoch 3500, Loss 2.927830
Epoch 4000, Loss 2.927679
Epoch 4500, Loss 2.927652
Epoch 5000, Loss 2.927647'''


# 优化器
import torch.optim as optim
# 每个优化器构造函数都将参数（通常是将require_grad设置为True的PyTorch张量）作为第一个输入。
# 传递给优化器的所有参数都保留在优化器对象内，以便优化器可以更新其值并访问grad属性

# 每个优化器有两个方法：zero_grad和step。
# zero_grad将构造时传递给优化器的所有参数的grad属性归零
# step根据特定优化器实施的优化策略更新这些参数的值
'''params = torch.tensor([1.0,0.0],requires_grad=True)
learning_rate = 1e-5
optimizer = optim.SGD([params],lr = learning_rate)'''  #SGd:stochastic gradient descent。这里优化器采用原始vanilla的梯度下降。只要动量momentum设置为默认值0.0
# “随机”（stochastic）来自以下事实：通常是通过平均输入样本的随机子集（称为minibatch）产生的梯度来获得最终梯度。
# 然而，优化器本身并不知道是对所有样本（vanilla）还是对其随机子集（stochastic）进行了损失评估，因此两种情况下的算法相同。
'''t_p = model(t_u,*params)
loss = loss_fn(t_p,t_c)
loss.backward()  #反向传播
optimizer.step()   #优化'''
# step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。
# 调用step发生的事情是：优化器通过将params减去learning_rate与grad的乘积来更新的params
# 执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度
# 要注意将梯度清零,若在循环中调用了前面的代码,则在每次调用backward时,梯度都会在叶节点中积累且传播
# 需要在调用backward之前插入额外的zero_grad

# print(params)   tensor([ 9.5483e-01, -8.2600e-04], requires_grad=True)
t_un = 0.1 * t_u  #n表示normalization
'''params = torch.tensor([1.0,0.0],requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params],lr = learning_rate)
t_p = model(t_un, *params)
loss = loss_fn(t_p,t_c)
optimizer.zero_grad()   #此调用可以在循环中更早的位置
loss.backward()
optimizer.step()'''

# print(params)  tensor([1.7761, 0.1064], requires_grad=True)

'''def training_loop(n_epochs,optimizer,params,t_u,t_c):
    for epoch in range(1,n_epochs+1):
        t_p = model(t_u,*params)
        loss = loss_fn(t_p,t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  #根据梯度值更新参数
        if epoch % 500 ==0:
            print('Epoch %d, Loss %f'%(epoch,float(loss)))
    return params'''

'''params = torch.tensor([1.0,0.0],requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params],lr=learning_rate)

training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    params = params,
    t_u = t_un,
    t_c=t_c)
'''

# Adam优化器:它自适应设置学习率,对参数缩放敏感度很低,可以使用原始(非标准化)输入t_u
'''params = torch.tensor([1.0,0.0],requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params],lr=learning_rate)

training_loop(
    n_epochs = 2000,
    optimizer = optimizer,
    params = params,
    t_u=t_u,
    t_c=t_c
)'''
'''Epoch 500, Loss 7.612900
Epoch 1000, Loss 3.086700
Epoch 1500, Loss 2.928579
Epoch 2000, Loss 2.927644'''


# 训练,验证,过拟合overfitting
# 你应确保为该过程获取了足够多的数据。如果你以很低频率从正弦过程中采样来收集数据，那么你很难让模型拟合这些数据
# 假设你有足够多的数据，则应确保能够拟合训练数据的模型在数据点之间尽可能正则化（regular）
# 方法:1.在损失函数中添加惩罚项,使模型的行为更平稳,变化更慢;2.向输入样本添加噪声,在训练数据样本之间人为创建新数据,迫使模型也尝试拟合它们
# 选择模型:选择正确大小的NN模型:增大模型大小直至成功拟合数据,再逐渐缩小直不再过拟合

# 对张量的元素进行打乱=重新排列其索引
n_samples = t_u.shape[0]
# print(n_samples)   11
n_val = int(0.2 * n_samples)  # Integer类型作用求不大于number 的最大整数   val验证集
# print(n_val)   2
shuffled_indices = torch.randperm(n_samples)
# print(shuffled_indices)   将0-10这11个数字随机打乱
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]
# print(train_indices,val_indices)   所得划分结果随机：tensor([ 4,  3,  8,  1,  7, 10,  0,  6,  2]) tensor([5, 9])

# 获得可用于从数据张量构建训练集和验证集的索引：
train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u
# 训练循环代码和之前一样，额外添加了评估每个epoch的验证损失以便查看是否过度拟合：
def training_loop(n_epochs,optimizer,params,train_t_u,val_t_u,train_t_c,val_t_c):
    for epoch in range(1,n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p,train_t_c)

        val_t_p = model(val_t_u,*params)
        val_loss = loss_fn(val_t_p,val_t_c)

        optimizer.zero_grad()
        train_loss.backward()  #没有val_backward()因为不能再验证集上训练模型
        optimizer.step()

        if epoch <=3 or epoch % 500 == 0:
            print('Epoch %d,Training loss %.2f,Validation loss %.2f'%(epoch,float(train_loss),float(val_loss)))
    return params

params = torch.tensor([1.0,0.0],requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params],lr = learning_rate)

training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    params = params,
    train_t_u = train_t_un,
    val_t_u = val_t_un,
    train_t_c = train_t_c,
    val_t_c = val_t_c
)

'''Epoch 1,Training loss 96.10,Validation loss 9.55
Epoch 2,Training loss 35.65,Validation loss 22.77
Epoch 3,Training loss 28.41,Validation loss 35.22
Epoch 500,Training loss 7.59,Validation loss 13.72
Epoch 1000,Training loss 3.97,Validation loss 6.11
Epoch 1500,Training loss 3.31,Validation loss 3.80
Epoch 2000,Training loss 3.20,Validation loss 2.99
Epoch 2500,Training loss 3.18,Validation loss 2.68
Epoch 3000,Training loss 3.17,Validation loss 2.55'''

# 验证集很小，验证损失仅在一定程度上有意义。

# 不需要时关闭autograd
# 训练循环中，只能在train_loss上调用backward，因此误差只会根据训练集来进行反向传播。验证集用于在未用于训练的数据上对模型输出的准确性进行独立评估
# 训练循环中的第一行在train_t_u上对模型进行评估以产生train_t_p。然后用train_t_p计算train_loss，创建一个链接从train_t_u到train_t_p再到train_loss的计算图。
# 当在val_t_u上再次评估模型然后生成val_t_p和val_loss时，将创建一个单独的计算图，该图链接从val_t_u到val_t_p再到val_loss。单独的张量通过相同的函数model和loss_fn运行，生成了单独的计算图。
# 如果还（错误地）对val_loss调用了backward，那么你将在相同叶节点上累积val_loss相对于参数的导数值。除非你明确地将梯度清零，否则每次调用backward时，梯度都会进行累积。
# 这里就会发生类似的事情：对val_loss调用backward会导致梯度累积在trainsloss.backward()执行期间生成的梯度之上。此时，你将在整个数据集（训练集加上验证集）上有效地训练模型，因为梯度将取决于两者。

# 通过使用torch.no_grad上下文管理器在不需要时关闭autograd——因为永远不会对val_loss调用backward，完全可以将model和loss_fn当作普通函数而无需追踪计算历史

def training_loop(n_epochs,optimizer,params,train_t_u,val_t_u,train_t_c,val_t_c):
    for epoch in range(1,n_epochs + 1):
        train_t_p = model(train_t_u,*params)
        train_loss = loss_fn(train_t_p,train_t_c)

        with torch.no_grad():
            val_t_p = model(val_t_u,*params)
            val_loss = loss_fn(val_t_p,val_t_c)
            assert val_loss.requires_grad == False

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
# 通过检查val_loss张量上require_grad属性的值确保上下文管理器正常工作：

# 使用相关的set_grad_enabled上下文管理器，你还可以根据布尔表达式（通常表示在训练还是在推理中）来调节代码在启用或禁用autograd的情况下运行。
def calc_forward(t_u,t_c,is_train):
    with torch.set_grad_enabled(is_train):   #根据布尔值is_train参数运行带或不带autograd的model和loss_fn
        t_p = model(t_u,*params)
        loss = loss_fn(t_p,t_c)
    return loss








