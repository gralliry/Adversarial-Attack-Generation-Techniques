# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 11:23
# @Author  : Liang Jinaye
# @File    : contrast_attack.py
# @Description :
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import torchvision
import matplotlib.pyplot as plt
import warnings

# FGSM, I_FGSM, MI_FGSM, L_BFGS, DeepFool, CW, JSMA, ONE_PIXEL, UPSET, ResidualModel
from attack import *

from models import ResNet18

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

classes = ('plane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def show(images, texts):
    # 创建一个4x1的子图布局
    fig, axes = plt.subplots(1, len(images))

    for i, image in enumerate(images):
        # 将张量转换为 NumPy 数组
        numpy_image = image.clone().detach().squeeze(dim=0).permute(1, 2, 0).cpu().numpy()

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

    dataloader = DataLoader(datasets, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    # 指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    model = ResNet18().to(device)
    model.load_state_dict(torch.load("./model/ResNet/train_100_0.9126999974250793.pth"))
    # (batch_size, 3, 32, 32) -> (batch_size, 10)

    residual_model = ResidualModel().to(device)
    residual_model.load_state_dict(torch.load("./model/UPSET/target_0/0.9653946161270142.pth"))
    print("预训练模型加载完成")

    # attacker = CW(model=model, criterion=criterion)
    # attacker = MI_FGSM(model=model, criterion=criterion)
    attacker = JSMA(model=model)

    print("攻击模型已创建")

    model.eval()
    for image, target in dataloader:
        image, target = image.to(device), target.to(device)
        output = model(image)

        attack_target = [(i + 1) % 10 for i in target]

        attack_image = attacker.attack(image, attack_target)
        attack_output = model(attack_image)

        print(output.argmax(1))
        print(attack_output.argmax(1))

        show(
            [image, attack_image],
            [
                f"True: {classes[target[0]]}  Predict: {classes[output.argmax(1)[0]]}",
                f"Attacked: {classes[attack_output.argmax(1)[0]]}",
            ]
        )

        input("任意输入继续生成")


if __name__ == "__main__":
    main()
