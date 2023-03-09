# 时间序列
# 华盛顿特区自行车共享系统中的数据，报告了2011年至2012年之间首都自行车共享系统中，\
# 租用自行车的每小时计数以及相应的天气和季节性信息。我们的目的是获取平面2D数据集并将其转换为3D数据集
import numpy as np
import torch
# torch.set_printoptions(profile="full")
bikes_numpy = np.loadtxt('./hour-fixed.csv', dtype=np.float32, delimiter=',', skiprows=1,
                         converters={1:lambda x:x[8:10]})   #转换第二列数据，取第二列数据第9和10位数字
bikes = torch.from_numpy(bikes_numpy)
'''
每个小时都会报告以下数据：
instant      # index of record #索引记录
day          # day of month #一个月中的某天
season       # season (1: spring, 2: summer, 3: fall, 4: winter) #季节（1：春天，2：夏天，3：秋天，4：冬天）
yr           # year (0: 2011, 1: 2012) #年份
mnth         # month (1 to 12) #月
hr           # hour (0 to 23) #小时
holiday      # holiday status #假期状态
weekday      # day of the week #一周的某天
workingday   # working day status #工作状态
weathersit   # weather situation #天气情况   (1: clear, 2:mist, 3: light rain/snow, 4: heavy rain/snow) #1：晴，2：薄雾，3：小雨/雪，4：大雨/雪
temp         # temperature in C #摄氏温度
atemp        # perceived temperature in C #感知温度（摄氏度）
hum          # humidity #湿度
windspeed    # windspeed #风速
casual       # number of causal users #因果用户数
registered   # number of registered users #注册用户数
cnt          # count of rental bikes #出租自行车数
'''
# print(bikes.size())    #torch.Size([17520, 17])
daily_bikes = bikes.view(-1,24,bikes.shape[1])
# bikes.shape[1] = 17，即取第二个数字
# print(daily_bikes.size)    torch.Size([730,24,17])
# print(daily_bikes.stride())
# 对于daily_bikes，步幅告诉你沿小时维度（第二个）前进1个位置需要你将存储（或一组列）中的位置前进17个位置，\
# 而沿日期维度（第一个）前进则需要你在时间24小时中前进等于行长度的元素数（此处为408，即17 * 24）。\
# 参数中的-1就代表这个位置由其他位置的数字来推断

# 将天气变量（四个等级）视为分类变量，转换为独热编码的向量。
# 1 暂时限制为第一天，初始化一个零填充的矩阵，其行数等于一天中的小时数，列数等于天气等级的数
first_day = bikes[:24].long()   #取第一天即24个小时的数据。即24行数据。size:([24,17])
weather_onehot = torch.zeros(first_day.shape[0],4)   #size=(24,4)
weather_onehot.scatter_(dim=1,index=first_day[:,9].unsqueeze(1)-1,value=1.0)   #unsqueeze表示升维，1表示在第二维的数据再升一维。unsqueeze(1)-1是指升维后的数字分别-1得到新的索引值.\
# dim=0 :按行填充，index的值为行的索引，列索引则由self tensor（此处即weather_oneshot）的列数决定、填写weather_shot有4列，则是0、1、2、3循环6次。\
# dim=1 :按列填充，index的值为列的索引，行的索引由weather_oneshot的行数决定，即0、1、2、3……23.
# first_day[:,9].shape:([24])       index = first_day[:,9].unsqueeze(1)-1.shape :   ([24,1])
# first_day[:,9].unsqueeze(1)-1   转置：tensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1]])

# print(weather_onehot)
# tensor([[1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 1., 0.],
#         [0., 1., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 1., 0., 0.]])

# 使用cat()将矩阵连接到原始数据集，看第一个结果
a = torch.cat((bikes[:24],weather_onehot),1)[:1]    #bikes[:24]:  shape:[24,17]  取前24行数据，即0-23行。
# weather_oneshot.shape: [24,4]     将两个矩阵按照1维进行拼接，即[24,17+4]     [:1] 取0行数据，查看第一个结果
# 最后的新四列分别是1,0,0,0——这正是你所期望的天气等级1

# 也可以使用重新排列的daily_bikes张量完成相同的操作。其形状为（B,C,L），L=24。\
# 首先创建零张量有相同的B和L，但增加的列数与C：

daily_bikes_onehot = torch.zeros(daily_bikes.shape[0],4,daily_bikes.shape[2])   #daily_bikes.shape = 730,24,17    daily_bikes.shape[0]=730, daily_bikes.shape[2]=17
# 沿c维度链接（）括号里第二个数字：
daily_bikes = torch.cat((daily_bikes,daily_bikes_onehot), dim=1)  # shape = torch.Size([730, 28, 17])

#这种方法并不是处理天气情况变量的唯一方法，其标签具有序数关系，因此可以暂时认为他们是连续变量的特殊值。可以转换变量，使其从0.0到1.0运行：
daily_bikes[:,9,:] = (daily_bikes[:,9,:]-1.0)/3.0    #不太懂为什么这么操作，可能是为了缩放变量

#有多种重新调整变量的方式，可以将其范围映射到[0.0,1.0]
temp = daily_bikes[:,10,:]
temp_min = torch.min(temp)    #输出temp中最小值: tensor(0.)
temp_max = torch.max(temp)
daily_bikes[:,10,:] = (daily_bikes[:,10,:] - temp_min) / (temp_max - temp_min)

# 或者减去平均值并除以标准差：
temp1 = daily_bikes[:,10,:]
daily_bikes[:,10,:] = (daily_bikes[:,10,:] - torch.mean(temp)) / torch.std(temp1)
# 该种情况下,变量的平均值为零,标准差为零.若取自高斯分布,则68%的样本将位于[-1.0,1.0]区间









