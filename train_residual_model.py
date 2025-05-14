# -*- coding: utf-8 -*-
# @Description: This is used to train the disturbance generation model required by the UPSET method
import argparse
import os.path

from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import attack
from models import IndentifyModel

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--target", required=True, type=int, choices=range(10), help="针对的target(0到9)")
parser.add_argument("-e", "--epoch", default=200, type=int, help="训练次数")
parser.add_argument("-b", "--batch_size", default=1024, type=int, help="Batch Size")
parser.add_argument("-lr", "--learning_rate", default=0.1, type=float, help="学习率")
parser.add_argument("-pim", "--path_indentify_model", required=True, type=str, help="识别模型路径")
parser.add_argument("-prm", "--path_residual_model", default="", type=str, help="残差模型路径")

args = parser.parse_args()


def main():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    # DataSet
    train_datasets = CIFAR10("./datasets", train=True, transform=transform_train)

    test_datasets = CIFAR10("./datasets", train=False, transform=transform_test)

    # DataLoader
    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, num_workers=14,
                                  drop_last=True)
    test_dataloader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False, num_workers=14,
                                 drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------Load the recognition model here-------------------
    right_model = IndentifyModel().to(device)
    right_model.load_state_dict(torch.load(args.path_indentify_model, weights_only=True, map_location=device))
    right_model.eval()

    # -------------------Load the UPSET interference model here-------------------
    residual_model = attack.ResidualModel().to(device)
    if args.path_residual_model:
        residual_model.load_state_dict(torch.load(args.path_residual_model, weights_only=True, map_location=device))

    optimizer = torch.optim.SGD(residual_model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    pardir = f"./parameter/UPSET/{args.target}"
    os.makedirs(pardir, exist_ok=True)
    #
    s = 0.1
    w = 2
    #
    for epoch in range(1, args.epoch + 1):
        residual_model.train()
        for images, _ in tqdm(train_dataloader, desc=f"Train:{epoch}/{args.epoch}"):
            images = images.to(device)
            attack_target = torch.tensor([args.target for _ in range(images.size(0))]).to(device)
            optimizer.zero_grad()

            attack_images = torch.clamp(s * residual_model(images) + images, 0, 1)
            attack_output = right_model(attack_images)

            loss1 = torch.nn.functional.cross_entropy(attack_output, attack_target, reduction='mean')
            loss2 = torch.nn.functional.mse_loss(attack_images, images)
            # print(loss1, loss2)
            loss = loss1 + w * loss2
            # print(loss1 / loss, loss2 / loss)

            loss.backward()
            optimizer.step()

        # Recording accuracy
        predict_accuracy = 0
        attacked_accuracy = 0
        total_loss = 0
        total_num = 0
        residual_model.eval()
        for images, targets in tqdm(test_dataloader, desc=f"Eval :{epoch}/{args.epoch}"):
            images, targets = images.to(device), targets.to(device)
            attack_images = torch.clamp(s * residual_model(images) + images, 0, 1)
            attack_output = right_model(attack_images)
            attack_target = torch.tensor([args.target for _ in range(images.size(0))]).to(device)

            predict_accuracy += targets.eq(attack_output.argmax(1)).sum().item()
            attacked_accuracy += attack_target.eq(attack_output.argmax(1)).sum().item()
            total_loss += torch.nn.functional.cross_entropy(attack_output, attack_target, reduction='mean').item()
            total_loss += w * torch.nn.functional.l1_loss(attack_images, images).item()
            total_num += images.size(0)

        scheduler.step()

        torch.save(residual_model.state_dict(), f"{pardir}/{attacked_accuracy / total_num:.7f}-{epoch}.pth")

        print(f"Identify success rate after predict: {predict_accuracy / total_num}")
        print(f"Identify error   rate after attack : {attacked_accuracy / total_num}")
        print(f"Test loss: {total_loss / total_num}")


if __name__ == "__main__":
    main()
