# -*- coding: utf-8 -*-
# @Description:
import os

import torch.optim
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import argparse

from models import *

parser = argparse.ArgumentParser()

parser.add_argument("-e", "--epoch", default=100, type=int, help="训练次数")

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
    # 准备数据集
    train_datasets = CIFAR10("./datasets", train=True, transform=transform_train)

    test_datasets = CIFAR10("./datasets", train=False, transform=transform_test)

    # 数据加载器
    train_dataloader = DataLoader(train_datasets, batch_size=256, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_datasets, batch_size=256, shuffle=False, num_workers=0)

    # 指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss().to(device)

    # 网络模型
    # https://github.com/kuangliu/pytorch-cifar

    # ------------------选择要训练的模型------------------
    # model = SimpleDLA()
    # model = VGG('VGG19')
    model = ResNet18()
    # model = PreActResNet18()
    # model = GoogLeNet()
    # model = DenseNet121()
    # model = ResNeXt29_2x64d()
    # model = MobileNet()
    # model = MobileNetV2()
    # model = DPN92()
    # model = ShuffleNetG2()
    # model = SENet18()
    # model = ShuffleNetV2(1)
    # model = EfficientNetB0()
    # model = RegNetX_200MF()
    # model = SimpleDLA()

    model = model.to(device)

    # ------------------这里可以加载已经训练过的模型参数文件来继续训练------------------
    model.load_state_dict(torch.load("./parameter/ResNet/train_100_0.9126999974250793.pth", map_location=device))

    model_name = model.__class__.__name__

    # 优化器
    """
    损失函数上，分别对神经网络参数的常见优化器SGD(stochastic gradient descent 随机梯度下降)、
    和Adam（Adaptive Moment Estimation 自适应矩估计）两种优化器进行了挑选，在实际效果上，发现SGD随机梯度下降对本实验的优化更好。
    """
    # 学习率
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 记录训练次数
    total_train_step = 0
    total_train_loss = 0

    if not os.path.exists(f"./tensorboard/{model_name}"):
        os.mkdir(f"./tensorboard/{model_name}")
    if not os.path.exists(f"./parameter/{model_name}"):
        os.mkdir(f"./parameter/{model_name}")
    # 训练过程记录器
    writer = SummaryWriter(f"./tensorboard/{model_name}")
    # 训练的轮数
    for i in range(args.epoch):
        print(f"第 {i + 1} 轮训练开始")
        model.train()
        for imgs, targets in train_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            output = model(imgs)

            loss = loss_fn(output, targets)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            total_train_loss += loss.item()

            if total_train_step % 100 == 0:
                print(f"训练次数: {total_train_step}, Loss: {total_train_loss / total_train_step}")
                writer.add_scalar("train_loss", total_train_loss / total_train_step, total_train_step)

        # 测试
        model.eval()
        total_test_loss = 0
        # 整体正确率
        total_num = 0
        total_accuracy = 0
        with torch.no_grad():
            for imgs, targets in test_dataloader:
                imgs, targets = imgs.to(device), targets.to(device)

                output = model(imgs)

                loss = loss_fn(output, targets)

                total_test_loss += loss.item()
                accuracy = (output.argmax(1) == targets).sum()

                total_num += test_dataloader.batch_size
                total_accuracy += accuracy

        # 记录第total_train_step次时的准确率和损失
        print(f"整体测试集上的loss: {total_test_loss / total_num}")
        writer.add_scalar("test_loss", total_test_loss / total_num, total_train_step)
        print(f"整体测试集上的正确率: {total_accuracy / total_num}")
        writer.add_scalar("test_accuracy", total_accuracy / total_num, total_train_step)

        # 保存训练参数文件
        torch.save(model.state_dict(), f"./parameter/{model_name}/train_{i + 1}_{total_accuracy / total_num}.pth")

        # 调整学习率
        scheduler.step()
    # tensorboard --logdir=tensorboard/{model_name} --port=6008
    writer.close()


if __name__ == "__main__":
    main()
