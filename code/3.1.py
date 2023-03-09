#表格数据
import torch
# torch.set_printoptions(profile="full")   #如果要让tensor完全显示
import csv
import numpy as np
wine_path = r'./winequality-white.csv'
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=';', skiprows=1)
col_list = next(csv.reader(open(wine_path), delimiter=';'))   #此处next()和csv.reader()一起是选取csv的第一行
wineq = torch.from_numpy(wineq_numpy)
data = wineq[:,:-1]  #除了最后一列外所有列
target = wineq[:,-1].long()   #最后一列quality
target_onehot = torch.zeros(target.shape[0],10)
target_onehot.scatter_(1,target.unsqueeze(1),1.0)

data_mean = torch.mean(data,dim=0)  #按列取平均数
data_var = torch.var(data,dim=0)
data_normalized = (data - data_mean)/torch.sqrt(data_var)
bad_indexes = torch.le(target,3)  #分辨坏酒，小于等于3
# print(bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum())
# torch.Size([4898]) torch.bool tensor(20),只有20个元素为1
# print(bad_indexes)  :tensor([False, False, False,  ..., False, False, False])
bad_data = data[bad_indexes]   #是一个张量，根据bad_indexes对data进行索引
# print(bad_data.shape)  torch.Size([20, 11])  只有20行，与bad_indexes张量1的个数相同。
mid_data = data[torch.gt(target, 3) & torch.lt(target, 7)]  #&“和”运算。   >3 & <7
good_data = data[torch.ge(target, 7)]   # >=7

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)
for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))
# *args:表示不知道参数个数      format()前面{}里，的数字表示精度，即显示的字符串/浮点数结果的长度。  6.2f：数字整体长度包括小数点为 6 位，保留 2 位小数，不足则以空格补齐，对齐方式为右对齐。

total_sulfur_threshold = 141.83  #使用二氧化硫总含量阈值作为区分好酒和怀酒的粗略标准
#获取二氧化硫总含量列中低于141.83的值
total_sulfur_data = data[:, 6]   #所有行，第6列
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
# print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())
# torch.Size([4898]) torch.bool tensor(2727)

#获取实际优质葡萄酒的索引
actual_indexes = torch.gt(target, 5)
# print(actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum())
# torch.Size([4898]) torch.bool tensor(3258)    可以发现实际优质酒比阈值预测的多了大约500例

#查看预测与实际的吻合程度。在预测索引和实际索引之间执行逻辑”与“运算得到交集
n_matches = torch.sum(actual_indexes & predicted_indexes).item()   #  &:同时为1，才得1，否则为0
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
print(n_matches, n_matches/n_predicted, n_matches/n_actual)