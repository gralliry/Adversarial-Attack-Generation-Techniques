# -*- coding: utf-8 -*-
# @Description: Select the optimal number of num workers by uptime
from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms

if __name__ == "__main__":
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./datasets/',
        train=True,
        # If true, the dataset is downloaded from the internet and placed in a directory.
        # If the dataset has already been downloaded, it will not be downloaded again.
        download=True,
        transform=transform
    )
    print(f"num of CPU: {mp.cpu_count()}")
    for num_workers in range(2, mp.cpu_count(), 2):
        train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=64,
                                                   pin_memory=True)
        start = time()
        for i, data in enumerate(train_loader, 0):
            pass
        end = time()
        print(f"Finish with:{end - start} second, num_workers={num_workers}")
