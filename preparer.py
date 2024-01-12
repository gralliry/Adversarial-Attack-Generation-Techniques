# -*- coding: utf-8 -*-
# @Time    : 2024/1/6 19:16
# @Author  : Liang Jinaye
# @File    : preparer.py
# @Description :
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models import *

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class Preparer:
    def __init__(self, cuda=True, **kwargs):
        # 指定设备
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.train_datasets = None
        self.test_datasets = None

        self.train_dataloader = None
        self.test_dataloader = None

        self.loss_fn = None

        self.model = None

        # 准备模型
        if model := kwargs.get('model', None) is not None:
            self.load_model(
                model=model,
                path=kwargs.get('path', None),
                is_state_dict=kwargs.get('is_state_dict', False)
            )

        # 准备数据集
        if train_datasets := kwargs.get('train_datasets', None) is not None:
            self.load_datasets(train_datasets=train_datasets)
        if test_datasets := kwargs.get('test_datasets', None) is not None:
            self.load_datasets(train_datasets=test_datasets)

        # 准备加载器
        if train_dataloader := kwargs.get('train_dataloader', None) is not None:
            self.load_dataloader(train_dataloader=train_dataloader)
        if test_dataloader := kwargs.get('test_dataloader', None) is not None:
            self.load_dataloader(test_dataloader=test_dataloader)

        # 准备损失函数
        if loss_fn := kwargs.get('loss_fn', None) is not None:
            self.load_loss_fn(loss_fn=loss_fn)

        self.train_datasets = CIFAR10("./datasets", train=True, transform=transform_train)

        self.test_datasets = CIFAR10("./datasets", train=False, transform=transform_test)

        # 数据加载器
        self.train_dataloader = DataLoader(self.train_datasets, batch_size=64, shuffle=True, num_workers=4)
        self.test_dataloader = DataLoader(self.test_datasets, batch_size=128, shuffle=False, num_workers=4)

        # 损失函数
        # 交叉熵损失函数
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def load_model(self, model, path=None, is_state_dict=False):
        if model is None:
            return
        if path is None:
            self.model = model.to(self.device)
        elif is_state_dict:
            self.model = model.to(self.device)
            self.model.load_state_dict(torch.load(path))
        else:
            self.model = torch.load(path)

    def load_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn.to(self.device)

    def load_datasets(self, train_datasets=None, test_datasets=None, **kwargs):
        if train_datasets is not None:
            self.train_datasets = train_datasets
            self.load_dataloader(train=True, **kwargs)
        if test_datasets is not None:
            self.test_datasets = test_datasets
            self.load_dataloader(train=False, **kwargs)

    def load_dataloader(self, train=True, **kwargs):
        if train and self.train_datasets is not None:
            self.train_dataloader = DataLoader(self.train_datasets, **kwargs)
        elif self.test_datasets is not None:
            self.test_dataloader = DataLoader(self.test_datasets, **kwargs)


preparer = Preparer(
    train_datasets=CIFAR10("./datasets", train=True, transform=transform_train),
    test_datasets=CIFAR10("./datasets", train=False, transform=transform_test),
    loss_fn=nn.CrossEntropyLoss(),

)
preparer.load_dataloader(
    train_datasets=CIFAR10("./datasets", train=True, transform=transform_train),
    test_datasets=CIFAR10("./datasets", train=False, transform=transform_test)
)
