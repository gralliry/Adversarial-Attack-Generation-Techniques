# -*- coding: utf-8 -*-
# @Description: Select the optimal number of num workers by uptime
import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms

if __name__ == "__main__":
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
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
    for num_workers in range(0, mp.cpu_count(), 2):
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=64,
                                                 pin_memory=True)
        start = time.time()
        for images, labels in dataloader:
            pass
        end = time.time()
        print(f"Finish with: {end - start} second, num_workers={num_workers}")
