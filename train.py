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

parser.add_argument("-e", "--epoch", default=200, type=int, help="Training times")
parser.add_argument("-b", "--batch_size", default=1024, type=int, help="Batch size")
parser.add_argument("-n", "--name", default="model", type=str, help="Name of model")
parser.add_argument("-p", "--path", default="", type=str, help="Path of model")
parser.add_argument("-lr", "--learning_rate", default=0.10, type=float, help="Learning rate")

args = parser.parse_args()


def main():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪+填充
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    # DataSet
    train_datasets = CIFAR10("./datasets", train=True, transform=transform_train)

    test_datasets = CIFAR10("./datasets", train=False, transform=transform_test)

    # DataLoader
    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, num_workers=14)
    test_dataloader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False, num_workers=14)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Recognition model (original model)
    model = IndentifyModel()
    model = model.to(device)

    if args.path != "":
        # Here you can load the already trained model parameter file to continue training
        model.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))

    name = args.name

    # Optimizer
    """
    Two optimizers were chosen in terms of the loss function，Adam and SGD。
    In practice, it was found that SGD stochastic gradient descent was more suitable.
    """
    # Optimizer Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    os.makedirs(f"./tensorboard/{name}", exist_ok=True)
    os.makedirs(f"./parameter/{name}", exist_ok=True)
    # Recorder
    writer = SummaryWriter(f"./tensorboard/{name}")
    # Number of training rounds
    for epoch in range(1, args.epoch + 1):
        model.train()
        train_num = 0
        train_loss = 0
        train_accuracy = 0
        for imgs, targets in tqdm(train_dataloader, desc=f"Train:{epoch}/{args.epoch}"):
            # print(torch.min(imgs), torch.max(imgs))
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            output = model(imgs)
            loss = loss_fn(output, targets)

            loss.backward()
            optimizer.step()

            train_num += imgs.size(0)
            train_loss += loss.item()
            train_accuracy += targets.eq(output.argmax(dim=1)).sum().item()

        # test
        model.eval()
        test_num = 0
        test_loss = 0
        test_accuracy = 0
        for imgs, targets in tqdm(test_dataloader, desc=f"Eval :{epoch}/{args.epoch}"):
            imgs, targets = imgs.to(device), targets.to(device)

            output = model(imgs)

            loss = loss_fn(output, targets)

            test_num += imgs.size(0)
            test_loss += loss.item()
            test_accuracy += targets.eq(output.argmax(dim=1)).sum().item()

        # Record the accuracy and loss of the total training step
        print(f"Train Loss    : {train_loss / train_num}")
        writer.add_scalar("train_loss", train_loss / train_num, epoch)
        print(f"Train Accuracy: {train_accuracy / train_num}")
        writer.add_scalar("test_accuracy", test_accuracy / test_num, epoch)
        print(f"Test  Loss    : {test_loss / test_num}")
        writer.add_scalar("test_loss", test_loss / test_num, epoch)
        print(f"Test  Accuracy: {test_accuracy / test_num}")
        writer.add_scalar("test_accuracy", test_accuracy / test_num, epoch)

        # Save the training parameter file
        torch.save(model.state_dict(), f"./parameter/{name}/{test_accuracy / test_num:.7f}-{epoch}.pth")

        # Adjust the learning rate
        scheduler.step()
    # tensorboard --logdir=tensorboard/{model_name} --port=6008
    writer.close()


if __name__ == "__main__":
    main()
