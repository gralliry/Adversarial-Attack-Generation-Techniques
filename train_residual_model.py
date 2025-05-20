# -*- coding: utf-8 -*-
# @Description: This is used to train the disturbance generation model required by the UPSET method
import argparse
import os.path

from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from attack.upset import ResidualModel
from models import IndentifyModel

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--target", default=-1, type=int, choices=range(-1, 10),
                    help="针对的target(-1,0~9)")
parser.add_argument("-e", "--epoch", default=200, type=int, help="Epoch")
parser.add_argument("-b", "--batch_size", default=512, type=int, help="Batch Size")
parser.add_argument("-lr", "--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("-pim", "--path_indentify_model", required=True, type=str, help="Path of indentify model")
parser.add_argument("-prm", "--path_residual_model", default="", type=str, help="Path of residual model")

args = parser.parse_args()


# CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python train_residual_model.py -pim parameter/ResNet18/0.pth -t -1
def criterion(epsilon, output, target, is_targeted):
    loss1 = torch.nn.functional.cross_entropy(output, target)
    if not is_targeted:
        loss1 = -loss1
    loss1 = 50 - 10 * loss1
    loss2 = 0.001 * torch.mean(torch.abs(epsilon))
    loss = loss1 + loss2
    return loss


def main():
    is_targeted = args.target != -1

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
    right_model.train()

    # -------------------Load the UPSET interference model here-------------------
    residual_model = ResidualModel().to(device)
    if args.path_residual_model:
        residual_model.load_state_dict(torch.load(args.path_residual_model, weights_only=True, map_location=device))

    optimizer = torch.optim.SGD(residual_model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    pardir = f"./parameter/UPSET/{args.target}"
    os.makedirs(pardir, exist_ok=True)
    #
    for epoch in range(1, args.epoch + 1):
        residual_model.train()
        for images, target in tqdm(train_dataloader, desc=f"Train:{epoch}/{args.epoch}"):
            images = images.to(device)
            if is_targeted:
                attack_target = torch.tensor([args.target for _ in range(images.size(0))], dtype=torch.long).to(device)
            else:
                attack_target = target.to(device)

            optimizer.zero_grad()

            epsilon = residual_model(images)
            attack_images = torch.clamp(epsilon + images, 0, 1)
            attack_output = right_model(attack_images)

            loss = criterion(epsilon, attack_output, attack_target, is_targeted)

            loss.backward()
            optimizer.step()

        # Recording accuracy
        accuracy = 0
        total_loss = 0
        total_num = 0
        residual_model.eval()
        for images, target in tqdm(test_dataloader, desc=f"Eval :{epoch}/{args.epoch}"):
            images = images.to(device)
            epsilon = residual_model(images)

            attack_images = torch.clamp(epsilon + images, 0, 1)
            attack_output = right_model(attack_images)

            if is_targeted:
                attack_target = torch.tensor([args.target for _ in range(images.size(0))], dtype=torch.long).to(device)
                accuracy += attack_target.eq(attack_output.argmax(1)).sum().item()
            else:
                attack_target = target.to(device)
                accuracy += attack_target.ne(attack_output.argmax(1)).sum().item()

            loss = criterion(epsilon, attack_output, attack_target, is_targeted)
            total_loss += loss.item()
            total_num += images.size(0)

        scheduler.step()

        torch.save(residual_model.state_dict(), f"{pardir}/{accuracy / total_num:.7f}-{epoch}.pth")

        print(f"Identify Error Rate after Attack: {accuracy / total_num}")
        print(f"Test loss: {total_loss / total_num}")


if __name__ == "__main__":
    main()
