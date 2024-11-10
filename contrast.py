# -*- coding: utf-8 -*-
# @Description:
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import matplotlib.pyplot as plt
import warnings

import argparse

# 识别模型
from models import IndentifyModel
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


def show(images, texts, is_show=False, is_save=True, save_path="./output.png"):
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
    if is_save:
        plt.savefig(save_path, dpi=300)
    # 展示图像
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

    # 在这里，您可以加载已训练的模型参数文件
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
        # 如果未选择此攻击方法，则可以忽略它
        residual_model = ResidualModel().to(device)
        # -------------------在此处加载 UPSET 干扰生成模型-------------------
        residual_model.load_state_dict(torch.load("./parameter/UPSET/target_0/0.pth"))
        # UPSET
        attacker = UPSET(model=residual_model)
    else:
        print(f"Unknown Method: {method}")
        return
    # ----------------------------------------------------------

    print("The attack model has been created")
    # 开始测试
    for index, (image, target) in enumerate(dataloader):
        image, target = image.to(device), target.to(device)

        origin_output = attacker.forward(image)
        print("Generating attack samples...")
        attack_image = attacker.attack(image, target)

        attack_output = model(attack_image)
        print("Generation complete.")
        # 使用 matplotlib 显示比较
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
