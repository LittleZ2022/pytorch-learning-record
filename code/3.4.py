# 图像数据
# 图像：按规则网格排列的标量集合，由高度和宽度（以像素为单位）。每个网格点（像素）可能只有一个标量，这种图像表示为灰度图像；每个王国点可能由多个标量，通常代表不同的颜色或不同的特征features。
# 代表单个像素值的标量通常使用8位整数编码。
# 常见的用数字编码颜色的方法：RGB，三个数字表示颜色（红、绿、蓝）。可将一个颜色通道视为仅讨论该颜色时的灰度强度图。

import imageio.v2 as imageio
import torch
import os
# 加载PNG图像，通过统一的API处理不同的数据类型
img_arr = imageio.imread('bobby.jpg')
# print(img_arr.shape)    (720,1280,3)
# 此时img_arr是一个numpy数组对象。三个维度：两个空间维度（宽、高），和对应RGB三个三色的第三维度。
# pytorch模块处理图像数据需要将张量设置为C * H * W（通道、高度、宽度）或转置transpose，交换第一个和最后一个通道获得正确的维数设置W * H * C
img = torch.from_numpy(img_arr)
out = torch.transpose(img,0,2)  #此操作不会复制张量数据。out与img使用相同的内部存储，只是修改了张量的尺寸和步幅信息

# 遵循与以前的数据类型相同的策略，创建包含多个图像的数据集以用作NN的输入，沿着第一维将这些图像按照批量存储，以获得N * C * H * W张量\
# 使用堆叠stack来构建张量。
# 1.预先分配适当尺寸的张量，从文件夹中加载图像填充它
batch_size = 100
batch = torch.zeros(100,3,256,256,dtype=torch.uint8) #100个RGB图像，分别为256像素高度和256像素宽度。张量的类型：每种颜色都以8位整数表示。
data_dir = './image-cats/'
filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)=='png']
for i ,filename in enumerate(filenames):
    img_arr = imageio.imread(filename)
    batch[i] = torch.transpose(torch.from_numpy(img_arr),0,2)
# NN通常使用浮点张量作为输入，当输入数据的范围大约为0-1或-1-1时，NN表现出最佳的训练性能
# 需要做的事情：将张量转换为浮点数并归一化像素值。归一化取决于决定的输入范围是0-1还是-1-1.
# 一种选择是将像素的值除以255（8位无符号最大可表示的数字）：
batch = batch.float()
batch /= 255.0
# 一种选择是计算输入数据的均值和标准偏差并对其进行缩放，以便于在每个通道上的均值和单位标准差=偏差输出为零：
n_channels = batch.shape[1]
# print(batch.shape) torch.Size([100, 3, 256, 256])
# print(range(n_channels))   range(0, 3)
for c in range(n_channels):
    mean = torch.mean(batch[:,c])   #c:channel
    std = torch.std(batch[:,c])
    batch[:,c] = (batch[:,c] - mean)/std













