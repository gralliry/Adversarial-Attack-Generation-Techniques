# -*- coding: utf-8 -*-
# @Description: 按运行时间选择最佳 num worker 数量
from time import time
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


if __name__ == "__main__":
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./datasets/',
        train=True,
        # 如果为 true，则从 Internet 下载数据集并将其放在目录中。
        # 如果数据集已下载，则不会再次下载。
        download=True,
        transform=transform
    )
    print(f"num of CPU: {mp.cpu_count()}")
    for num_workers in range(2, mp.cpu_count(), 2):
        train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=num_workers, batch_size=64,
                                                   pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
