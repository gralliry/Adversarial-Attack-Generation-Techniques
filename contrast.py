# -*- coding: utf-8 -*-
# @Description:
import os

import torch
import torchvision
import matplotlib.pyplot as plt
import warnings

import argparse

from torchvision import transforms
# Identify model
from models import IndentifyModel
# Adversarial model
from attack import FGSM, I_FGSM, MI_FGSM, L_BFGS, DeepFool, CW, JSMA, ONE_PIXEL, UPSET, ResidualModel

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method',
                    required=True,
                    choices=['L-BFGS', 'FGSM', 'I-FGSM', 'JSMA', 'ONE-PIXEL', 'CW', 'DEEPFOOL', 'MI-FGSM', 'UPSET'],
                    help="Test method: L-BFGS, FGSM, I-FGSM, JSMA, ONE-PIXEL, CW, DEEPFOOL, MI-FGSM, UPSET")
parser.add_argument('-p', '--path', required=True, help="The path of the model parameter file")
parser.add_argument('-t', '--target', type=int, default=-1, help="The target of attacking if it is targeted")
parser.add_argument('-os', '--only_success', action="store_true", default=False,
                    help="Only successful images will be output")
parser.add_argument('-or', '--only_right', action="store_true", default=False,
                    help="Only right images will be output")
parser.add_argument('-sst', '--skip_same_target', action="store_true", default=False,
                    help="Skip the same target if is_targeted")

args = parser.parse_args()

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

classes = ('plane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# python contrast.py -p parameter/ResNet18/0.pth -m FGSM


def show(images, texts, is_show=False, is_save=True, save_path="./output.png"):
    # Create a 4x1 subgraph layout
    fig, axes = plt.subplots(1, len(images))

    for i, image in enumerate(images):
        # Show the images
        axes[i].imshow(image)
        axes[i].set_title(texts[i])

    # Adjust the layout to avoid overlapping
    plt.tight_layout()
    if is_save:
        plt.savefig(save_path, dpi=300)
    # Show the images
    if is_show:
        plt.show()
    plt.close()


def main():
    is_targeted = args.target > -1

    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10("./datasets", train=False, transform=transform)

    # batch_size must be 1
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Here, you can load the trained model parameter file
    model = IndentifyModel().to(device)
    model.load_state_dict(torch.load(args.path, map_location=device, weights_only=True))

    print("The Pre-Training Model is Loaded")
    # ----------------------------------------------------------
    method = args.method.upper()
    if method == "L-BFGS":
        attacker = L_BFGS(model=model, epsilon=0.01, alpha=0.1, iters=10, lr=0.001)
    elif method == "FGSM":
        attacker = FGSM(model=model, epsilon=0.01)
    elif method == "I-FGSM":
        attacker = I_FGSM(model=model, epsilon=0.001, alpha=0.01, iters=15)
    elif method == "JSMA":
        attacker = JSMA(model=model, theta=0.05, gamma=0.1)
    elif method == "ONE-PIXEL":
        attacker = ONE_PIXEL(model=model, pixels_size=100, pixels_changed=10)
    elif method == "CW":
        attacker = CW(model=model, c=1, kappa=1, steps=50, lr=0.01)
    elif method == "DEEPFOOL":
        attacker = DeepFool(model=model, overshoot=0.01, iters=10)
    elif method == "MI-FGSM":
        attacker = MI_FGSM(model=model, alpha=0.01, decay=0.3, iters=10)
    elif method == "UPSET":
        # Disturbance generation model
        residual_model = ResidualModel().to(device)
        # -------------------Load the UPSET interference generation model here-------------------
        residual_model.load_state_dict(torch.load(f"./parameter/UPSET/0/0.pth", weights_only=True, map_location=device))
        # UPSET
        attacker = UPSET(model=model, residual_model=residual_model)
    else:
        raise ValueError(f"Unknown Method: {method}")
    # ----------------------------------------------------------
    os.makedirs(f"./output/{method}", exist_ok=True)
    print("The Attack Model has been Created")
    # Start testing
    num = 0
    num_try = 0
    for image, target in dataloader:
        num_try += 1
        image, target = image.to(device), target.to(device)
        attack_target = torch.full_like(target, args.target) if is_targeted else target

        if args.skip_same_target and is_targeted:
            if target == attack_target:
                continue

        origin_output = attacker.forward(image)
        print(f"Generating Attack Samples...Try: {num_try}", end="\r")

        attack_image = attacker.attack(image, attack_target, is_targeted=is_targeted)

        attack_output = model(attack_image)

        if args.only_right:
            if origin_output.argmax(1) != target:
                continue

        if args.only_success:
            if is_targeted:
                if attack_output.argmax(1) != attack_target:
                    continue
            else:
                if attack_output.argmax(1) == attack_target:
                    continue

        # Comparisons are displayed using matplotlib
        image_showed = image.detach().cpu()[0].permute(1, 2, 0).numpy()
        attack_image_showed = attack_image.detach().cpu()[0].permute(1, 2, 0).numpy()
        show(
            [
                image_showed,
                attack_image_showed,
            ], [
                f"True: {classes[target[0]]}\n"
                f"Predict: {classes[origin_output.argmax(1)[0]]}",
                f"Expect: {'' if is_targeted else 'not '}{classes[attack_target[0]]}\n"
                f"Attacked: {classes[attack_output.argmax(1)[0]]}",
            ],
            is_show=False,
            is_save=True, save_path=f"./output/{method}/{num}.png"
        )
        num += 1
        num_try = 0

        print("\nCompleted!")
        input("Enter Any to Continue Generating...")


if __name__ == "__main__":
    main()
