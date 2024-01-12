# -*- coding: utf-8 -*-
# @Time    : 2024/1/6 19:00
# @Author  : Liang Jinaye
# @File    : test_attack.py
# @Description :
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets

from attack import *

from models import ResNet18


def main():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # 测试数据集
    test_datasets = datasets.CIFAR10("./datasets", train=False, transform=transform_test)
    # 数据加载器
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    # 指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    # 识别模型
    model = ResNet18().to(device)
    model.load_state_dict(torch.load("./model/ResNet/train_100_0.9126999974250793.pth"))

    # model = models.vgg19().to(device)
    # model.load_state_dict(torch.load("./model/pretrained/checkpoints/vgg19-dcbb9e9d.pth"))

    print("预训练模型加载完成")

    # -------------------------------------------
    method = "deepfool"
    method = method.upper()
    if method == "TEST":
        # TEST
        attacker = BaseModel(model=model)
    elif method == "L-BFGS":
        # L-BFGS
        attacker = L_BFGS(model=model, criterion=criterion)
        # attacker = L_BFGS(model=model, criterion=criterion, iters=2, epsilon=0.2)
    elif method == "FGSM":
        # FGSM
        attacker = FGSM(model=model, criterion=criterion)
        # attacker = FGSM(model=model, criterion=criterion, epsilon=0.2)
    elif method == "I-FGSM":
        # I-FGSM
        attacker = I_FGSM(model=model, criterion=criterion)
        # attacker = I_FGSM(model=model, criterion=criterion)
    elif method == "JSMA":
        # JSMA
        attacker = JSMA(model=model)
        # attacker = JSMA(model=model, alpha=6, gamma=6, iters=50)
    elif method == "ONE-PIXEL":
        # ONE-PIXEL
        attacker = ONE_PIXEL(model=model)
        # attacker = ONE_PIXEL(model=model)
    elif method == "C&W":
        # C&W
        attacker = CW(model=model, criterion=criterion)
        # attacker = CW(model=model, criterion=criterion, iters=1000)
    elif method == "DEEPFOOL":
        # DEEPFOOL
        attacker = DeepFool(model=model)
        # attacker = DeepFool(model=model, overshoot=2, iters=100)
    elif method == "MI-FGSM":
        # MI-FGSM
        attacker = MI_FGSM(model=model, criterion=criterion)
        # attacker = MI_FGSM(model=model, criterion=criterion)
    elif method == "UPSET":
        # UPSET
        residual_model = ResidualModel().to(device)
        residual_model.load_state_dict(torch.load("./model/UPSET/target_0/0.9653946161270142.pth"))
        attacker = UPSET(model=residual_model)
        # attacker = UPSET(model=residual_model)
    else:
        print(f"Unknown Method: {method}")
        return
    # -------------------------------------------
    # 开始测试
    attacker.test_attack(model=model, dataloader=test_dataloader, max_counter=500)


if __name__ == "__main__":
    main()
