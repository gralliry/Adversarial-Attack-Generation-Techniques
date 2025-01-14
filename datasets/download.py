# -*- coding: utf-8 -*-
# @Description:
from torchvision import datasets

DOWNLOAD = True

# Download the dataset
train_datasets = datasets.CIFAR10("./", train=True, download=DOWNLOAD)

test_datasets = datasets.CIFAR10("./", train=False, download=DOWNLOAD)
