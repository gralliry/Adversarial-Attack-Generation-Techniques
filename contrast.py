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
# 运行时输入需要测试的方法
parser.add_argument('-m', '--method',
                    required=True,
                    choices=['L-BFGS', 'FGSM', 'I-FGSM', 'JSMA', 'ONE-PIXEL', 'C&W', 'DEEPFOOL', 'MI-FGSM', 'UPSET'],
                    help="Test method: L-BFGS, FGSM, I-FGSM, JSMA, ONE-PIXEL, C&W, DEEPFOOL, MI-FGSM, UPSET")

args = parser.parse_args()
# 忽略特定类型的警告
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

    # 指定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # 识别模型
    # -------------------请在这里加载识别模型-------------------
    model = ResNet18().to(device)
    model.load_state_dict(torch.load("./parameter/ResNet/train_100_0.9126999974250793.pth"))

    print("预训练模型加载完成")
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
        # 扰动生成模型
        # -------------------请在这里加载UPSET扰动模型-------------------（如果不是选用这个攻击方法，可以忽视）
        residual_model = ResidualModel().to(device)
        residual_model.load_state_dict(torch.load("./parameter/UPSET/target_0/0.9653946161270142.pth"))
        # UPSET
        attacker = UPSET(model=residual_model)
    else:
        print(f"Unknown Method: {method}")
        return
    # ----------------------------------------------------------

    print("攻击模型已创建")

    model.eval()
    # 开始测试
    for image, target in dataloader:
        image, target = image.to(device), target.to(device)
        output = model(image)

        # 生成攻击标签
        # ----------这里只是单纯错开正确标签，可以换成想要的攻击标签----------
        # attack_target = [0 for i in target]  # 这个就是对第一个标签0即plane进行攻击
        attack_target = [(i + 1) % 10 for i in target]

        print("正在生成攻击样本...")
        # 生成对抗样本
        # ----------可以自行添加参数（如果有）来调整攻击效果----------
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
        print("生成完成.")
        # 使用matplotlib展示对比
        show(
            [image, attack_image],
            [
                f"True: {classes[target[0]]}  Predict: {classes[output.argmax(1)[0]]}",
                f"Attacked: {classes[attack_output.argmax(1)[0]]}",
            ]
        )

        input("任意输入回车继续生成...")


if __name__ == "__main__":
    main()
