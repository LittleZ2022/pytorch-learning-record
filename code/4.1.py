# 学习就是参数估计
# 给定输入数据和相应的期望输出（ground truth）以及权重的初始值，模型输入数据（前向传播），然后通过把结果输出与ground truth进行比较来评估误差。
# 为了优化模型的参数，其权重（即单位权重变化引起的误差变化，也即误差相对于参数的梯度）通过使用对复合函数求导的链式法则进行计算（反向传播）。
# 权重的值沿导致误差减小的方向更新。不断重复该过程直到在新数据上的评估误差降至可接受的水平以下。

# 问题：假设你去了一些鲜为人知的地方旅游，然后带回了一个花哨的壁挂式模拟温度计。这个温度计看起来很棒，非常适合你的客厅。
# 唯一的缺点是它不显示单位。
# 用某种单位建立一个读数和相应温度值的数据集，选择一个模型，并迭代调整单位的权重，直到误差的测量值足够低为止，最后可以在新温度计上进行准确读数。
import torch
# 1.记录能正常工作的旧摄氏温度计的数据+新温度计对应的测量值。将数据转换维张量
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)    #摄氏度
t_u = torch.tensor(t_u)    #未知单位度数

# 2.假设一个用于两组测量之间相互转换的最简单的可能模型，即线性模型
def model(t_u,w,b):
    return w * t_u + b
#t_u：输入张量，w: weight权重参数, b:bias偏置参数。
# 该模型中，参数是pytorch标量（即零维张量），乘积运算将使用broadcast来产生返回的张量
# 损失函数：计算训练样本的期望输出与模型接受这些样本所产生的实际输出之间的差异
def loss_fn(t_p,t_c):   #t_p:预测温度，t_c：实际测量值
    squared_diffs = (t_p-t_c)**2
    return squared_diffs.mean()

# 建立一个张量差，将它们按元素elementwise取平方，通过对所得张量中的所有元素求平均来得到标量损失函数。这个损失是平方误差损失
# 3.初始化参数，调用模型
w = torch.ones(1)
b = torch.zeros(1)
t_p = model(t_u,w,b)
# print(t_p)
# tensor([35.7000, 55.9000, 58.2000, 81.9000, 56.3000, 48.9000, 33.9000, 21.8000,
#         48.4000, 60.4000, 68.4000])

# 4.计算损失函数的值
loss = loss_fn(t_p,t_c)
# print(loss)
# tensor(1763.8848)

# 5.梯度下降:计算相对于每个参数的损失变化率，并沿损失减小的方向改变每个参数。
# 通过对w和b进行小改动来估计变化率，以查看该邻域中的损失值的变化
# model里得到更新参数w的预测值，与实际值t_c的loss_fn
# 在w和b的当前值附近很小的范围内，w的单位增加会导致损失的某些变化。如果变化为负，则需要增加w以使损失最小，而如果变化为正，则需要减小w。
delta = 0.1
loss_rate_of_change_w = (loss_fn(model(t_u,w + delta,b),t_c) -
                         loss_fn(model(t_u,w - delta,b),t_c)) / (2.0 * delta)
# 通常缓慢地更改参数，因为损失变化率可能会与当前w值的距离相差很大。
# 因此，用一个较小的因子来缩小变化率。这个因子有很多名称，机器学习中称之为学习率（learning rate）。
learning_rate = 1e-2
w = w - learning_rate * loss_rate_of_change_w

# 对b进行同样的操作
loss_rate_of_change_b = (loss_fn(model(t_u,w,b+delta),t_c)-
                         loss_fn(model(t_u,w,b-delta),t_c))/(2.0 * delta)
b = b - learning_rate * loss_rate_of_change_b
# 以上是梯度下降的基本参数更新步骤。通过不断重复这个步骤（假设你选择的学习率足够低），收敛到参数的最佳值，对于该参数，根据给定数据计算出的损失最小。
# 但是这种计算变化率的方法相当粗糙，通过重复评估模型和损失以搜索w和b附近的损失函数的行为来计算变化率，并不能很好地适应具有许多参数的模型。
# 此外，并不总是很清楚该邻域应该有多大。之前选择的delta等于0.1，但是一切都取决于损失函数的尺寸。如果损失与delta相比变化得太快，那么你将对下降后的位置一无所知。

# 损失函数的导数表达式:
def dloss_fn(t_p,t_c):
    dsq_diffs = 2 * (t_p - t_c)
    return dsq_diffs
# 对w求导：
def dmodel_dw(t_u,w,b):
    return t_u
# 对d求导：
def dmodel_db(t_u,w,b):
    return 1.0
# 返回损失相对于w和b的梯度的函数
def grad_fn(t_u, t_c, t_p, w ,b):
    dloss_dw = dloss_fn(t_p,t_c) * dmodel_dw(t_u,w,b)
    dloss_db = dloss_fn(t_p,t_c) * dmodel_dw(t_u,w,b)
    return torch.stack([dloss_dw.mean(),dloss_db.mean()])  #默认dim为0


# 训练循环。
# 进行参数优化，从参数的暂定值开始，迭代地对其应用更新以进行固定次数的迭代或直到w和b停止改变位置。
# 以固定迭代次数为例：
# epoch:所有训练样本上的依次参数更新迭代
def training_loop(n_epochs, learning_rate, params, t_u, t_c, print_params = True, verbose=1):  #verbose日志显示
    for epoch in range(1, n_epochs+1):
        w,b = params

        t_p = model(t_u,w,b)  #前向传播，得到预测值
        loss = loss_fn(t_p,t_c)
        grad = grad_fn(t_u,t_c,t_p,w,b)  #反向传播

        params = params - learning_rate * grad

        if epoch % verbose == 0:   #求模运算，相当于mod，也就是计算除法的余数，比如5%2就得到1。
            print('Epoch %d, Loss %f' % (epoch, float(loss)))   # %格式化输出
            if print_params:
                print('  Params: ',params)
                print('  Grad  :' , grad)
    return params
# 调用上述训练循环函数：
'''training_loop(
    n_epochs = 10,
    learning_rate = 1e-2,
    params = torch.tensor([1.0,0.0]),
    t_u = t_u,
    t_c = t_c
)'''

'''Epoch 1, Loss 1763.884766
  Params:  tensor([-44.1730, -45.1730])
  Grad  : tensor([4517.2964, 4517.2964])
Epoch 2, Loss 6008401.500000
  Params:  tensor([2614.3445, 2613.3445])
  Grad  : tensor([-265851.7500, -265851.7500])
Epoch 3, Loss 20810780672.000000
  Params:  tensor([-153844.6250, -153845.6250])
  Grad  : tensor([15645897., 15645897.])
Epoch 4, Loss 72079231680512.000000
  Params:  tensor([9054074., 9054073.])
  Grad  : tensor([-9.2079e+08, -9.2079e+08])
Epoch 5, Loss 249650368558923776.000000
  Params:  tensor([-5.3285e+08, -5.3285e+08])
  Grad  : tensor([5.4190e+10, 5.4190e+10])
Epoch 6, Loss 864677195443788054528.000000
  Params:  tensor([3.1359e+10, 3.1359e+10])
  Grad  : tensor([-3.1892e+12, -3.1892e+12])
Epoch 7, Loss 2994855705791727814049792.000000
  Params:  tensor([-1.8456e+12, -1.8456e+12])
  Grad  : tensor([1.8769e+14, 1.8769e+14])
Epoch 8, Loss 10372844443041696866877046784.000000
  Params:  tensor([1.0861e+14, 1.0861e+14])
  Grad  : tensor([-1.1046e+16, -1.1046e+16])
Epoch 9, Loss 35926898610785416711310078377984.000000
  Params:  tensor([-6.3922e+15, -6.3922e+15])
  Grad  : tensor([6.5008e+17, 6.5008e+17])
Epoch 10, Loss 124434741885831511754115810712354816.000000
  Params:  tensor([3.7619e+17, 3.7619e+17])
  Grad  : tensor([-3.8258e+19, -3.8258e+19])'''

# 结果显示：参数params的更新太大；它们的值在每次更新过头时开始来回摆动，而下次更新则摆动更剧烈。
# 优化过程不稳定，它发散了而不是收敛到最小值。希望看到越来越小的参数更新，而不是越来越大，
# 因此要限制learning_rate * grad的大小
'''training_loop(
    n_epochs = 10,
    learning_rate = 1e-4,
    params = torch.tensor([1.0,0.0]),
    t_u = t_u,
    t_c = t_c
)'''

'''Epoch 1, Loss 1763.884766
  Params:  tensor([ 0.5483, -0.4517])
  Grad  : tensor([4517.2964, 4517.2964])
Epoch 2, Loss 307.417969
  Params:  tensor([ 0.3669, -0.6331])
  Grad  : tensor([1813.6058, 1813.6058])
Epoch 3, Loss 72.340569
  Params:  tensor([ 0.2941, -0.7059])
  Grad  : tensor([728.1271, 728.1271])
Epoch 4, Loss 34.322933
  Params:  tensor([ 0.2649, -0.7351])
  Grad  : tensor([292.3288, 292.3288])
Epoch 5, Loss 28.144279
  Params:  tensor([ 0.2531, -0.7469])
  Grad  : tensor([117.3642, 117.3642])
Epoch 6, Loss 27.128002
  Params:  tensor([ 0.2484, -0.7516])
  Grad  : tensor([47.1194, 47.1194])
Epoch 7, Loss 26.956022
  Params:  tensor([ 0.2465, -0.7535])
  Grad  : tensor([18.9174, 18.9174])
Epoch 8, Loss 26.925013
  Params:  tensor([ 0.2458, -0.7542])
  Grad  : tensor([7.5950, 7.5950])
Epoch 9, Loss 26.918699
  Params:  tensor([ 0.2455, -0.7545])
  Grad  : tensor([3.0492, 3.0492])
Epoch 10, Loss 26.917147
  Params:  tensor([ 0.2453, -0.7547])
  Grad  : tensor([1.2242, 1.2242])'''
# 结果显示参数更新很小，因此损失会缓慢下降最终停滞，通过动态调整learning_rate解决
# 在更新过程中另一个潜在的麻烦是：梯度本身。
# 例子中第一个epoch的params的w和b相差50倍，权重和偏差存在于不同比例的空间中。
# 足够大的学习率足以有意义地更新一个，但对于另一个不稳定，或者适合第二个学习者的学习率不足以有意义地更新第一个。
# 此时除非你更改问题的描述，否则你将无法更新参数。你可以为每个参数设置单独的学习率，但是对于具有许多参数的模型，此方法将非常麻烦。
# 可以采用一种更简单的方法：更改输入使梯度差别不要太大。粗略地说，你可以确保输入不要与范围-1.0到1.0相差过大。
# 可以通过将t_u乘以0.1来达到此效果：
t_un = 0.1 * t_u  #n表示normalization
'''training_loop(
    n_epochs = 10,
    learning_rate = 1e-4,
    params = torch.tensor([1.0,0.0]),
    t_u = t_un,
    t_c = t_c
)'''
'''Epoch 1, Loss 80.364342
  Params:  tensor([1.0078, 0.0078])
  Grad  : tensor([-77.6140, -77.6140])
Epoch 2, Loss 79.681824
  Params:  tensor([1.0155, 0.0155])
  Grad  : tensor([-77.0771, -77.0771])
Epoch 3, Loss 79.008888
  Params:  tensor([1.0231, 0.0231])
  Grad  : tensor([-76.5440, -76.5440])
Epoch 4, Loss 78.345390
  Params:  tensor([1.0307, 0.0307])
  Grad  : tensor([-76.0145, -76.0145])
Epoch 5, Loss 77.691231
  Params:  tensor([1.0383, 0.0383])
  Grad  : tensor([-75.4886, -75.4886])
Epoch 6, Loss 77.046249
  Params:  tensor([1.0458, 0.0458])
  Grad  : tensor([-74.9664, -74.9664])
Epoch 7, Loss 76.410316
  Params:  tensor([1.0532, 0.0532])
  Grad  : tensor([-74.4478, -74.4478])
Epoch 8, Loss 75.783340
  Params:  tensor([1.0606, 0.0606])
  Grad  : tensor([-73.9328, -73.9328])
Epoch 9, Loss 75.165161
  Params:  tensor([1.0680, 0.0680])
  Grad  : tensor([-73.4214, -73.4214])
Epoch 10, Loss 74.555672
  Params:  tensor([1.0752, 0.0752])
  Grad  : tensor([-72.9135, -72.9135])

'''

# 运行循环进行足够的迭代次数以查看参数的变化：
params = training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_un,
    t_c = t_c,
    print_params = False,
    verbose=500)
'''Epoch 500, Loss 32.665413
Epoch 1000, Loss 32.665413
Epoch 1500, Loss 32.665413
Epoch 2000, Loss 32.665413
Epoch 2500, Loss 32.665413
Epoch 3000, Loss 32.665413
Epoch 3500, Loss 32.665413
Epoch 4000, Loss 32.665413
Epoch 4500, Loss 32.665413
Epoch 5000, Loss 32.665413'''
# something weird，Loss很大，与例子完全不同

# 数据绘图
# %matplotlib inline   #是IPython的内置magic函数，那么在Pycharm中是不会支持的。%matplotlib inline 可以在Ipython编译器里直接使用，功能是可以内嵌绘图，并且可以省略掉plt.show()这一步。

from matplotlib import pyplot as plt

t_p = model(t_un, *params)  #“*”:用params收集传入是不定个数的参数，并将收集的到参数以元组的方式存储在params中，若无传入params就是个空元组

fig = plt.figure(dpi = 600)   #dpi:每英寸点数
plt.xlabel('Fahrenheit')
plt.ylabel('Celsius')

plt.plot(t_u.numpy(),t_p.detach().numpy())  #在原数据上作图
plt.plot(t_u.numpy(),t_c.numpy(),'o')
plt.show()
# 结果并不理想