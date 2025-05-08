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
    # DataSet
    train_datasets = CIFAR10("./datasets", train=True, transform=transform_train)

    test_datasets = CIFAR10("./datasets", train=False, transform=transform_test)

    # DataLoader
    train_dataloader = DataLoader(train_datasets, batch_size=1024, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_datasets, batch_size=1024, shuffle=False, num_workers=0)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Recognition model (original model)
    model = IndentifyModel()

    model = model.to(device)

    # Here you can load the already trained model parameter file to continue training
    # model.load_state_dict(torch.load("./parameter/ResNet/train_100_0.9126999974250793.pth", map_location=device))

    model_name = model.__class__.__name__

    # Optimizer
    """
    Two optimizers were chosen in terms of the loss function，Adam and SGD。
    In practice, it was found that SGD stochastic gradient descent was more suitable for the optimization of this experiment.
    """
    # Learning rate
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # Use Cosine Annealing to adjust the learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    os.makedirs(f"./tensorboard/{model_name}", exist_ok=True)
    os.makedirs(f"./parameter/{model_name}", exist_ok=True)
    # Recorder
    writer = SummaryWriter(f"./tensorboard/{model_name}")
    # Number of training rounds
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

        # Record the accuracy and loss of the total training step
        print(f"train loss: {train_loss / train_num}")
        writer.add_scalar("train_loss", train_loss / train_num, epoch)
        print(f"test loss: {test_loss / test_num}")
        writer.add_scalar("test_loss", test_loss / test_num, epoch)
        print(f"test accuracy: {test_accuracy / test_num}")
        writer.add_scalar("test_accuracy", test_accuracy / test_num, epoch)

        # Save the training parameter file
        torch.save(model.state_dict(), f"./parameter/{model_name}/{epoch}.pth")

        # Adjust the learning rate
        scheduler.step()
    # tensorboard --logdir=tensorboard/{model_name} --port=6008
    writer.close()


if __name__ == "__main__":
    main()
