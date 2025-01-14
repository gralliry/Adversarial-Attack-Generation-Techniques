# -*- coding: utf-8 -*-
# @Description:
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import matplotlib.pyplot as plt
import warnings

import argparse

# Identify model
from models import IndentifyModel
# Adversarial model
from attack import FGSM, I_FGSM, MI_FGSM, L_BFGS, DeepFool, CW, JSMA, ONE_PIXEL, UPSET, ResidualModel

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method',
                    required=True,
                    choices=['L-BFGS', 'FGSM', 'I-FGSM', 'JSMA', 'ONE-PIXEL', 'C&W', 'DEEPFOOL', 'MI-FGSM', 'UPSET'],
                    help="Test method: L-BFGS, FGSM, I-FGSM, JSMA, ONE-PIXEL, C&W, DEEPFOOL, MI-FGSM, UPSET")

args = parser.parse_args()

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

classes = ('plane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def show(images, texts, is_show=False, is_save=True, save_path="./output.png"):
    # Create a 4x1 subgraph layout
    fig, axes = plt.subplots(1, len(images))

    for i, image in enumerate(images):
        # Convert tensors to NumPy arrays
        numpy_image = torch.clamp(image, 0, 1).detach().squeeze(dim=0).permute(1, 2, 0).cpu().numpy()

        # Show the images
        axes[i].imshow(numpy_image)
        axes[i].set_title(texts[i])

    # Adjust the layout to avoid overlapping
    plt.tight_layout()
    if is_save:
        plt.savefig(save_path, dpi=300)
    # Show the images
    if is_show:
        plt.show()


def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    datasets = torchvision.datasets.CIFAR10("./datasets", train=False, transform=transform)

    dataloader = torch.utils.data.dataloader.DataLoader(datasets, batch_size=1, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Here, you can load the trained model parameter file
    model = IndentifyModel().to(device)
    model.load_state_dict(torch.load("./parameter/ResNet/100.pth"))

    print("The pre-training model is loaded")
    # ----------------------------------------------------------
    method = args.method.upper()
    if method == "L-BFGS":
        # L-BFGS
        attacker = L_BFGS(model=model, criterion=criterion)
    elif method == "FGSM":
        # FGSM
        attacker = FGSM(model=model, criterion=criterion)
    elif method == "I-FGSM":
        # I-FGSM
        attacker = I_FGSM(model=model, criterion=criterion, epsilon=1)
    elif method == "JSMA":
        # JSMA
        attacker = JSMA(model=model)
    elif method == "ONE-PIXEL":
        # ONE-PIXEL
        # attacker = ONE_PIXEL(parameter=parameter)
        attacker = ONE_PIXEL(model=model, pixels_changed=10)
    elif method == "C&W":
        # C&W
        attacker = CW(model=model, criterion=criterion)
    elif method == "DEEPFOOL":
        # DEEPFOOL
        # attacker = DeepFool(parameter=parameter)
        attacker = DeepFool(model=model, overshoot=2, iters=100)
    elif method == "MI-FGSM":
        # MI-FGSM
        attacker = MI_FGSM(model=model, criterion=criterion)
    elif method == "UPSET":
        # Disturbance generation model
        # If this attack method is not chosen, it can be ignored
        residual_model = ResidualModel().to(device)
        # -------------------Load the UPSET interference generation model here-------------------
        residual_model.load_state_dict(torch.load("./parameter/UPSET/target_0/0.pth"))
        # UPSET
        attacker = UPSET(model=residual_model)
    else:
        print(f"Unknown Method: {method}")
        return
    # ----------------------------------------------------------

    print("The attack model has been created")
    # Start testing
    for index, (image, target) in enumerate(dataloader):
        image, target = image.to(device), target.to(device)

        origin_output = attacker.forward(image)
        print("Generating attack samples...")
        attack_image = attacker.attack(image, target)

        attack_output = model(attack_image)
        print("Generation complete.")
        # Comparisons are displayed using matplotlib
        show(
            [
                image,
                attack_image
            ], [
                f"True: {classes[target[0]]}  Predict: {classes[origin_output.argmax(1)[0]]}",
                f"Attacked: {classes[attack_output.argmax(1)[0]]}",
            ],
            is_show=False,
            is_save=True, save_path=f"./output/{attacker.__class__.__name__}/{index}.png"
        )

        input("Enter any press enter to continue generating...")


if __name__ == "__main__":
    main()
