# -*- coding: utf-8 -*-
# @Description: Select the optimal number of num workers by uptime
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
        # If true, the dataset is downloaded from the internet and placed in a directory.
        # If the dataset has already been downloaded, it will not be downloaded again.
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
