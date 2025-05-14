# -*- coding: utf-8 -*-
# @Description:
import argparse
import warnings

from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets

from attack import *

from models import IndentifyModel

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', required=True,
                    choices=['L-BFGS', "FGSM", 'I-FGSM', 'JSMA', 'ONE-PIXEL', 'CW', 'DEEPFOOL', 'MI-FGSM', 'UPSET'],
                    help="Test method: L-BFGS, FGSM, I-FGSM, JSMA, ONE-PIXEL, CW, DEEPFOOL, MI-FGSM, UPSET")
parser.add_argument('-c', '--count', default=1000, type=int,
                    help="Number of tests (default is 500), but if the number of test datasets is less than this "
                         "number, the number of test datasets prevails")
parser.add_argument("-p", "--path", required=True, type=str, help="Path of model")
args = parser.parse_args()


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10("./datasets", train=False, transform=transform)

    # There are some methods that support batch_size is not 1,
    # just set it according to the method, if you don't know, then keep 1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IndentifyModel().to(device)
    # -------------------------------------------
    # Here, you can load the trained model parameter file
    # Once loaded, you can delete the warning
    model.load_state_dict(torch.load(args.path, map_location=device, weights_only=True))

    print("The Pre-Trained Model is Loaded")

    # -------------------------------------------
    method = args.method.upper()
    if method == "L-BFGS":
        attacker = L_BFGS(model=model)
    elif method == "FGSM":
        attacker = FGSM(model=model)
    elif method == "I-FGSM":
        attacker = I_FGSM(model=model)
    elif method == "JSMA":
        attacker = JSMA(model=model)
    elif method == "ONE-PIXEL":
        attacker = ONE_PIXEL(model=model)
    elif method == "CW":
        attacker = CW(model=model)
    elif method == "DEEPFOOL":
        attacker = DeepFool(model=model)
    elif method == "MI-FGSM":
        attacker = MI_FGSM(model=model)
    elif method == "UPSET":
        residual_model = ResidualModel().to(device)
        warnings.warn("You Must Load The Parameter of UPSET Model")
        # residual_model.load_state_dict(torch.load("./parameter/UPSET/0/1.pth"))
        attacker = UPSET(model=model, residual_model=residual_model)
    else:
        raise ValueError(f"Unknown Method: {method}")
    # -------------------------------------------
    # begin to test
    counter = 0
    max_counter = min(args.count, len(dataloader))
    print(f"Total Test Num: {max_counter}")
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
        total_num += image.size(0)
        attack_accuracy = target.eq(attack_output.argmax(1)).sum().item()
        origin_accuracy = target.eq(orinal_output.argmax(1)).sum().item()

        total_origin_accuracy += origin_accuracy
        total_attack_accuracy += attack_accuracy

        tqdm_dataloader.set_postfix(AttackAcc=f"{attack_accuracy / image.size(0)}",
                                    OriginAcc=f"{origin_accuracy / image.size(0)}")

        if tqdm_dataloader.n >= max_counter:
            break

    print(f"{attacker.__class__.__name__} "
          f"Initial      Accuracy Rate: {total_origin_accuracy / total_num} "
          f"After-attack Accuracy Rate: {total_attack_accuracy / total_num} ")


if __name__ == "__main__":
    main()
