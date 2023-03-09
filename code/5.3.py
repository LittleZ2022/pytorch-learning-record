# nn的子类
# 第一种nn.Sequential
import torch.nn as nn
import torch
seq_model = nn.Sequential(
            nn.Linear(1, 11),
            nn.Tanh(),
            nn.Linear(11, 1))
# print(seq_model)

# 输出:
# Sequential(
#   (0): Linear(in_features=1, out_features=11, bias=True)
#   (1): Tanh()
#   (2): Linear(in_features=11, out_features=1, bias=True)
# )

# 使用有序字典而不是列表作为输入为每一层添加标签：
from collections import OrderedDict
namedseq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 12)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(12 , 1))
]))
# print(namedseq_model)

# 输出：
# Sequential(
#   (hidden_linear): Linear(in_features=1, out_features=12, bias=True)
#   (hidden_activation): Tanh()
#   (output_linear): Linear(in_features=12, out_features=1, bias=True)
# )


# 除了`nn.Sequential`类提供的顺序性之外，你不能控制通过网络的数据流向。你可以自己定义`nn.Module`的子类来完全控制输入数据的处理方式
class SubclassModel(nn.Module):
    def __init__(self):
        super().__init__()  #super().__init__() 调用父类的init方法
        self.hidden_linear = nn.Linear(1,13)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(13,1)
    def forward(self,input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
        return output_t
subclass_model = SubclassModel()
# print(subclass_model)
#
# 输出:
# SubclassModel(
#   (hidden_linear): Linear(in_features=1, out_features=13, bias=True)
#   (hidden_activation): Tanh()
#   (output_linear): Linear(in_features=13, out_features=1, bias=True)
# )
# 该类的打印输出类似于具有命名参数的顺序模型`namedseq_model`的打印输出。因为使用了相同的名称并打算实现相同的网络结构。

# for type_str, model in [('seq', seq_model), ('namedseq', namedseq_model),
#      ('subclass', subclass_model)]:
#     print(type_str)
#     for name_str, param in model.named_parameters():
#         print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
#     print()

# 输出
# 0.weight              torch.Size([11, 1]) 11
# 0.bias                torch.Size([11])    11
# 2.weight              torch.Size([1, 11]) 11
# 2.bias                torch.Size([1])     1
#
# namedseq
# hidden_linear.weight  torch.Size([12, 1]) 12
# hidden_linear.bias    torch.Size([12])    12
# output_linear.weight  torch.Size([1, 12]) 12
# output_linear.bias    torch.Size([1])     1
#
# subclass
# hidden_linear.weight  torch.Size([13, 1]) 13
# hidden_linear.bias    torch.Size([13])    13
# output_linear.weight  torch.Size([1, 13]) 13
# output_linear.bias    torch.Size([1])     1

# 调用`named_parameters()`会深入搜寻构造函数中分配为属性的所有子模块，然后在这些子模块上递归调用`named_parameters()`。
# 无论子模块如何嵌套，任何`nn.Module`实例都可以访问其所有子参数的列表。通过访问将由`autograd`计算出的`grad`属性，优化器就知道如何更新参数以最大程度地减少损失
# Python列表或dict实例中包含的子模块不会被自动登记,可以使用[`add_module(name, module)`]



class SubclassFunctionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 14)
        # 去掉了nn.Tanh()
        self.output_linear = nn.Linear(14, 1)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = torch.tanh(hidden_t)  # nn.Tanh对应的函数
        output_t = self.output_linear(activated_t)
        return output_t


func_model = SubclassFunctionalModel()
# print(func_model)

# 输出:
# SubclassFunctionalModel(
#   (hidden_linear): Linear(in_features=1, out_features=14, bias=True)
#   (output_linear): Linear(in_features=14, out_features=1, bias=True)
# )
