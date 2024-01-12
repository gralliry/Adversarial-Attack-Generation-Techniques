# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 15:26
# @Author  : Liang Jinaye
# @File    : download.py
# @Description :
from torchvision import datasets

DOWNLOAD = True

# 准备数据集
train_datasets = datasets.CIFAR10("./", train=True, download=DOWNLOAD)

test_datasets = datasets.CIFAR10("./", train=False, download=DOWNLOAD)
