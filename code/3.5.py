# 体积数据
# 在诸如涉及CT（Computed Tomography）扫描等医学成像应用程序的情况下，通常需要处理从头到脚方向堆叠的图像序列，每个序列对应于整个身体的横截面。\
# 在CT扫描中，强度代表身体不同部位的密度：肺、脂肪、水、肌肉、骨骼，以密度递增的顺序排列，当在临床工作站上显示CT扫描时，会从暗到亮映射。\
# 根据穿过人体后到达检测器的X射线量计算每个点的密度，并使用一些复杂的数学运算将原始传感器数据反卷积（deconvolve）为完整体积数据。
# CT具有单个的强度通道，这类似于灰度图像。通常在本地数据格式中，通道维度被忽略了，因此原始数据通常会具有三个维度。\
# 通过将单个2D切片堆叠到3D张量中，你可以构建表示对象的3D解剖结构的体积数据。

# 在通道（channel）维之后，你有一个额外的维——深度（depth），形成5D张量为N x C x D x H x W。

# 加载一个CT扫描样本。将所有DICOM(digital imaging communication and storage)系列文件组合成一个numpy3D数组
import imageio
import torch

dir_path = './volumetric-dicom/2-LUNG 3.0  B70f-04083'    #dir:directory
vol_arr = imageio.volread(dir_path,'DICOM')
# print(vol_arr.shape)    \
# Reading DICOM (examining files): 1/99 files (1.0%99/99 files (100.0%)   Found 1 correct series.
# Reading DICOM (loading data): 87/99  (87.999/99  (100.0%)
# (99, 512, 512)

# 缺少通道信息，因此要新增通道channel维
vol = torch.from_numpy(vol_arr).float()
vol = torch.transpose(vol,0,2)
vol = torch.unsqueeze(vol,0)
# print(vol.shape)
# Reading DICOM (examining files): 99/99 files (100.0%)
#   Found 1 correct series.
# Reading DICOM (loading data): 99/99  (100.0%)
# torch.Size([1, 512, 512, 99])

