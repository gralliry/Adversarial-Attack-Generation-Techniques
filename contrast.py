# -*- coding: utf-8 -*-
# @Description:
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import matplotlib.pyplot as plt
import warnings

import argparse

# 识别模型
from models import ResNet18
# 对抗模型
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


def show(images, texts):
    # 创建一个4x1的子图布局
    fig, axes = plt.subplots(1, len(images))

    for i, image in enumerate(images):
        # 将张量转换为 NumPy 数组
        numpy_image = torch.clamp(image, 0, 1).detach().squeeze(dim=0).permute(1, 2, 0).cpu().numpy()

        # 展示图像
        axes[i].imshow(numpy_image)
        axes[i].set_title(texts[i])

    # 调整布局，避免重叠
    plt.tight_layout()
    # 展示图像
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

    # Here you can load the already trained model parameter file
    model = ResNet18().to(device)
    model.load_state_dict(torch.load("./parameter/ResNet/train_100_0.9126999974250793.pth"))

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
        # -------------------Load the UPSET disturbance model here-------------------
        # If this attack method is not selected, it can be ignored
        residual_model = ResidualModel().to(device)
        residual_model.load_state_dict(torch.load("./parameter/UPSET/target_0/0.9653946161270142.pth"))
        # UPSET
        attacker = UPSET(model=residual_model)
    else:
        print(f"Unknown Method: {method}")
        return
    # ----------------------------------------------------------

    print("The attack model has been created")

    model.eval()
    # 开始测试
    for image, target in dataloader:
        image, target = image.to(device), target.to(device)
        output = model(image)

        # Generate attack tag
        # Here you simply stagger the correct labels and replace them with the attack labels you want
        attack_target = [(i + 1) % 10 for i in target]
        # attack_target = [0 for i in target]  # This is an attack on the first tag 0, plane

        print("Generating attack samples...")
        # Generate adversarial sample
        # You can add your own parameters (if any) to adjust the attack effect
        if method == "L-BFGS":
            attack_image = attacker.attack(image, attack_target)
        elif method == "FGSM":
            attack_image = attacker.attack(image, target)
        elif method == "I-FGSM":
            attack_image = attacker.attack(image, target)
        elif method == "JSMA":
            attack_image = attacker.attack(image, attack_target)
        elif method == "ONE-PIXEL":
            attack_image = attacker.attack(image, target, is_targeted=False)
        elif method == "C&W":
            attack_image = attacker.attack(image, attack_target)
        elif method == "DEEPFOOL":
            attack_image = attacker.attack(image)
        elif method == "MI-FGSM":
            attack_image = attacker.attack(image, target)
        elif method == "UPSET":
            attack_image = attacker.attack(image)
        else:
            print(f"Unknown Method: {method}")
            return
        # ------------------------------------

        attack_output = model(attack_image)
        print("Generation complete.")
        # Show the comparison using matplotlib
        show(
            [image, attack_image],
            [
                f"True: {classes[target[0]]}  Predict: {classes[output.argmax(1)[0]]}",
                f"Attacked: {classes[attack_output.argmax(1)[0]]}",
            ]
        )

        input("Enter any press enter to continue generating...")


if __name__ == "__main__":
    main()
