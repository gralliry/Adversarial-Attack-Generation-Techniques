# -*- coding: utf-8 -*-
# @Description:
import argparse
import warnings

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets

from attack import *

from models import IndentifyModel

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', required=True,
                    choices=['L-BFGS', "FGSM", 'I-FGSM', 'JSMA', 'ONE-PIXEL', 'C&W', 'DEEPFOOL', 'MI-FGSM', 'UPSET'],
                    help="Test method: L-BFGS, FGSM, I-FGSM, JSMA, ONE-PIXEL, C&W, DEEPFOOL, MI-FGSM, UPSET")
parser.add_argument('-c', '--count', default=1000, type=int,
                    help="Number of tests (default is 500), but if the number of test datasets is less than this "
                         "number, the number of test datasets prevails")
args = parser.parse_args()


def main():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_datasets = datasets.CIFAR10("./datasets", train=False, transform=transform_test)

    # There are some methods that support batch_size is not 1, just set it according to the method, if you don't know, then keep 1
    dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss().to(device)

    model = IndentifyModel().to(device)
    # -------------------------------------------
    # Here, you can load the trained model parameter file
    # warnings.warn(f"You Must Load The Parameter of Model: {model.__class__.__name__}")
    # Once loaded, you can delete the warning
    model.load_state_dict(torch.load(f"./parameter/{model.__class__.__name__}/100.pth"))

    print("The pretrained model is loaded")

    # -------------------------------------------
    method = args.method.upper()
    if method == "L-BFGS":
        # L-BFGS
        attacker = L_BFGS(model=model, criterion=criterion)
        # attacker = L_BFGS(parameter=parameter, criterion=criterion, iters=2, epsilon=0.2)
    elif method == "FGSM":
        # FGSM
        attacker = FGSM(model=model, criterion=criterion)
        # attacker = FGSM(parameter=parameter, criterion=criterion, epsilon=0.2)
    elif method == "I-FGSM":
        # I-FGSM
        attacker = I_FGSM(model=model, criterion=criterion)
        # attacker = I_FGSM(parameter=parameter, criterion=criterion)
    elif method == "JSMA":
        # JSMA
        attacker = JSMA(model=model)
        # attacker = JSMA(parameter=parameter, alpha=6, gamma=6, iters=50)
    elif method == "ONE-PIXEL":
        # ONE-PIXEL
        attacker = ONE_PIXEL(model=model)
        # attacker = ONE_PIXEL(parameter=parameter)
    elif method == "C&W":
        # C&W
        attacker = CW(model=model, criterion=criterion)
        # attacker = CW(parameter=parameter, criterion=criterion, iters=1000)
    elif method == "DEEPFOOL":
        # DEEPFOOL
        attacker = DeepFool(model=model)
        # attacker = DeepFool(parameter=parameter, overshoot=2, iters=100)
    elif method == "MI-FGSM":
        # MI-FGSM
        attacker = MI_FGSM(model=model, criterion=criterion)
        # attacker = MI_FGSM(parameter=parameter, criterion=criterion)
    elif method == "UPSET":
        # UPSET
        residual_model = ResidualModel().to(device)
        warnings.warn(f"You Must Load The Parameter of Model: {residual_model.__class__.__name__}")
        # residual_model.load_state_dict(torch.load("./parameter/UPSET/target_0/1.pth"))
        attacker = UPSET(model=residual_model)
    else:
        print(f"Unknown Method: {method}")
        return
    # -------------------------------------------
    # begin to test
    counter = 0
    max_counter = min(args.count, len(dataloader))
    print(f"Total Test Num: {max_counter}")
    batch_size = dataloader.batch_size
    # Overall accuracy
    total_num = 0
    total_origin_accuracy = 0
    total_attack_accuracy = 0

    model.eval()
    # This is based on the minimum number of max_count and datasets you set
    tqdm_dataloader = tqdm(dataloader, desc="Test", total=max_counter)
    for image, target in tqdm_dataloader:
        # Update the progress bar
        image, target = image.to(device), target.to(device)

        # Initial results (not attacked)
        orinal_output = attacker.forward(image)

        # Generate an attack image
        pert_image = attacker.attack(image, target)
        # post-attack result
        attack_output = attacker.forward(pert_image)

        counter += 1
        total_num += batch_size
        attack_accuracy = (attack_output.argmax(1) == target).sum()
        origin_accuracy = (orinal_output.argmax(1) == target).sum()

        total_origin_accuracy += origin_accuracy
        total_attack_accuracy += attack_accuracy

        tqdm_dataloader.set_postfix(AttackAcc=f"{attack_accuracy / batch_size}",
                                    OriginAcc=f"{origin_accuracy / batch_size}")

        if tqdm_dataloader.n >= max_counter:
            break

    print(f"{attacker.__class__.__name__} "
          f"Initial      accuracy rate: {total_origin_accuracy / (max_counter * batch_size)} "
          f"After-attack accuracy rate: {total_attack_accuracy / (max_counter * batch_size)} ")


if __name__ == "__main__":
    main()
