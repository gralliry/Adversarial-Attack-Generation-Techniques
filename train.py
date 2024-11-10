# -*- coding: utf-8 -*-
# @Description:
import os
from tqdm import tqdm

import torch.optim
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import argparse

from models import IndentifyModel

parser = argparse.ArgumentParser()

parser.add_argument("-e", "--epoch", default=100, type=int, help="Training times")

args = parser.parse_args()


def main():
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
    # 数据集
    train_datasets = CIFAR10("./datasets", train=True, transform=transform_train)

    test_datasets = CIFAR10("./datasets", train=False, transform=transform_test)

    # 数据加载器
    train_dataloader = DataLoader(train_datasets, batch_size=1024, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_datasets, batch_size=1024, shuffle=False, num_workers=0)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # 识别模型（原本的模型）
    # https://github.com/kuangliu/pytorch-cifar

    model = IndentifyModel()

    model = model.to(device)

    # Here you can load the already trained model parameter file to continue training
    # model.load_state_dict(torch.load("./parameter/ResNet/train_100_0.9126999974250793.pth", map_location=device))

    model_name = model.__class__.__name__

    # 优化器
    """
    在损失函数方面选择了两个优化器，Adam （Adaptive Moment Estimation） 和 SGD （Adaptive Moment Estimation）。
    在实践中，发现 SGD 随机梯度下降更适合该实验的优化。
    """
    # 学习率
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # 使用 Cosine Annealing余弦退火 调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    os.makedirs(f"./tensorboard/{model_name}", exist_ok=True)
    os.makedirs(f"./parameter/{model_name}", exist_ok=True)
    # 训练过程记录器
    writer = SummaryWriter(f"./tensorboard/{model_name}")
    # 训练轮数
    for epoch in range(1, args.epoch + 1):
        model.train()
        train_num = 0
        train_loss = 0
        for imgs, targets in tqdm(train_dataloader, desc=f"Train:{epoch}/{args.epoch}"):
            imgs, targets = imgs.to(device), targets.to(device)
            output = model(imgs)

            loss = loss_fn(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_num += train_dataloader.batch_size
            train_loss += loss.item()

        print(f"Train Epoch: {epoch}, Loss: {train_loss / train_num}")
        writer.add_scalar("train_loss", train_loss / train_num, epoch)

        # test
        model.eval()
        test_num = 0
        test_loss = 0
        test_accuracy = 0
        for imgs, targets in tqdm(test_dataloader, desc=f"Eval:{epoch}/{args.epoch}"):
            imgs, targets = imgs.to(device), targets.to(device)

            output = model(imgs)

            loss = loss_fn(output, targets)

            test_num += test_dataloader.batch_size
            test_loss += loss.item()
            test_accuracy += (output.argmax(1) == targets).sum()

        # 记录总训练步长的精度和损失
        print(f"test loss: {test_loss / test_num}")
        writer.add_scalar("test_loss", test_loss / test_num, epoch)
        print(f"test accuracy: {test_accuracy / test_num}")
        writer.add_scalar("test_accuracy", test_accuracy / test_num, epoch)

        # 保存训练参数文件
        torch.save(model.state_dict(), f"./parameter/{model_name}/{epoch}.pth")

        # 调整学习率
        scheduler.step()
    # tensorboard --logdir=tensorboard/{model_name} --port=6008
    writer.close()


if __name__ == "__main__":
    main()
