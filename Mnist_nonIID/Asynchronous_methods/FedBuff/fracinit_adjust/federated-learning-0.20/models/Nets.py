#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, BatchNorm2d, Dropout


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNEmnist(nn.Module):
    def __init__(self, args):
        super(CNNEmnist, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=args.num_channels,  # 输入通道数
                out_channels=16,  # 输出通道数
                kernel_size=5,   # 卷积核大小
                stride=1,  #卷积步数
                padding=2,  # 如果想要 con2d 出来的图片长宽没有变化, 
                            # padding=(kernel_size-1)/2 当 stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, args.num_classes)  # 全连接层，A/Z,a/z一共37个类

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output
    

class Conv_Block(Module):
    
    def __init__(self, inchannel, outchannel, res=True):
        super(Conv_Block, self).__init__()
        self.res = res  # 是否带残差连接
        self.left = Sequential(
            Conv2d(inchannel, outchannel, kernel_size=(3, 3), padding=1, bias=False),
            BatchNorm2d(outchannel),
            ReLU(inplace=True),
            Conv2d(outchannel, outchannel, kernel_size=(3, 3), padding=1, bias=False),
            BatchNorm2d(outchannel),
        )
        self.shortcut = Sequential(Conv2d(inchannel, outchannel, kernel_size=(1,1), bias=False),
                                   BatchNorm2d(outchannel))
        self.relu = Sequential(
            ReLU(inplace=True))

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class Res_Model(Module):
    def __init__(self, res=True):
        super(Res_Model, self).__init__()

        self.block1 = Conv_Block(inchannel=3, outchannel=64)
        self.block2 = Conv_Block(inchannel=64, outchannel=128)
        self.block3 = Conv_Block(inchannel=128, outchannel=256)
        self.block4 = Conv_Block(inchannel=256, outchannel=512)
        # 构建卷积层之后的全连接层以及分类器：

        self.classifier = Sequential(Flatten(),  # 7 Flatten层
                                     Dropout(0.4),
                                     Linear(2048, 256),  # 8 全连接层
                                     Linear(256, 64),  # 8 全连接层
                                     Linear(64, 10))  # 9 全连接层 )   # fc，最终Cifar10输出是10类

        self.relu = ReLU(inplace=True)
        self.maxpool = Sequential(MaxPool2d(kernel_size=2))  # 1最大池化层

    def forward(self, x):
        out = self.block1(x)
        out = self.maxpool(out)
        out = self.block2(out)
        out = self.maxpool(out)
        out = self.block3(out)
        out = self.maxpool(out)
        out = self.block4(out)
        out = self.maxpool(out)
        out = self.classifier(out)
        return out