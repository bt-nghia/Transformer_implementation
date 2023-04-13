import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self, mul):
        super(MyModule, self).__init__()
        self.mul = mul

    def forward(self, x):
        x = x * self.mul
        return x


class MyModuleList(nn.Module):
    def __init__(self):
        super(MyModuleList, self).__init__()
        self.modList = nn.ModuleList([MyModule(mul=_) for _ in range(1,5)])

    def forward(self, x):
        for layer in self.modList:
            x = layer(x)
        return x


x = torch.tensor([1.0])
mylist = MyModuleList()
output = mylist(x)
print(output)
