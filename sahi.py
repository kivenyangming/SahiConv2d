import torch.nn.functional as F
import torch
import torch.nn
from torch import nn


class SahiConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(SahiConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # 生成随机的sahi mask，大小和输入张量一样
        sahi_mask = torch.randint_like(x, high=2)
        sahi_mask = sahi_mask.type(torch.float32)

        # 将sahi mask应用到输入张量中
        x = x * sahi_mask

        # 使用卷积层对sahi操作后的输入张量进行卷积
        x = self.conv(x)

        return x