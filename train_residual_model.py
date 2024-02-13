# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 13:10
# @Author  : Jianye Liang
# @File    : train_residual_model.py
# @Description : 这是用来训练UPSET方法需要的扰动生成模型
import os.path

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import argparse

from models import ResNet18

from attack import ResidualModel

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--target", required=True, type=int, choices=range(10), help="针对的target(0到9)")
parser.add_argument("-e", "--epoch", default=100, type=int, help="训练次数")
parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float, help="学习率")

args = parser.parse_args()


def main():
    attack_target = args.target

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
    # 准备数据集
    train_datasets = CIFAR10("./datasets", train=True, transform=transform_train)

    test_datasets = CIFAR10("./datasets", train=False, transform=transform_test)

    # 数据加载器
    train_dataloader = DataLoader(train_datasets, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    test_dataloader = DataLoader(test_datasets, batch_size=128, shuffle=False, num_workers=0, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------请在这里加载识别模型-------------------
    right_model = ResNet18().to(device)
    right_model.load_state_dict(torch.load("./parameter/ResNet/train_100_0.9126999974250793.pth"))
    right_model.eval()

    residual_model = ResidualModel().to(device)
    # residual_model.load_state_dict(torch.load("./parameter/UPSET/target_0/0.9653946161270142.pth"))

    loss_fn = nn.CrossEntropyLoss().to(device)

    learning_rate = args.learning_rate
    optimizer = torch.optim.SGD(residual_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 这里针对标签值
    attack_targets = torch.tensor([attack_target for _ in range(train_dataloader.batch_size)], device=device)

    # 记录正确率
    attacked_accuracy = 0
    predict_accuracy = 0
    total_num = 0
    for i in range(args.epoch):
        print(f"第 {i + 1} 轮训练开始")
        residual_model.train()
        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)

            attack_images = residual_model(images) + images
            attack_output = right_model(attack_images)

            loss = loss_fn(attack_output, attack_targets)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        residual_model.eval()
        with torch.no_grad():
            for images, targets in test_dataloader:
                images, targets = images.to(device), targets.to(device)
                attack_images = residual_model(images) + images
                attack_output = right_model(attack_images)

                predict_accuracy += (attack_output.argmax(1) == targets).sum()
                attacked_accuracy += (attack_output.argmax(1) == attack_targets).sum()
                total_num += test_dataloader.batch_size

        # 调整学习率
        scheduler.step()

        if not os.path.exists(f"./parameter/UPSET/target_{attack_target}"):
            os.makedirs(f"./parameter/UPSET/target_{attack_target}")

        torch.save(residual_model.state_dict(),
                   f"./parameter/UPSET/target_{attack_target}/{attacked_accuracy / total_num}.pth")

        print(f"预测后识别成功率: {predict_accuracy / total_num}")
        print(f"攻击后识别错误率: {attacked_accuracy / total_num}")


if __name__ == "__main__":
    main()
