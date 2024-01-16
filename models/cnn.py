# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 18:30
# @Author  : Jianye Liang
# @File    : cnn.py
# @Description :
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.neuralnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.neuralnet(x)
        return x
