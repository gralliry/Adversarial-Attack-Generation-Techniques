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
parser.add_argument('-c', '--count', default=500, type=int,
                    help="Number of tests (default is 500), but if the number of test datasets is less than this "
                         "number, the number of test datasets prevails")
args = parser.parse_args()


def main():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_datasets = datasets.CIFAR10("./datasets", train=False, transform=transform_test)

    dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss().to(device)

    model = IndentifyModel().to(device)
    # -------------------------------------------
    # Here you can load the already trained model parameter file
    warnings.warn(f"You Must Load The Parameter of Model: {model.__class__.__name__}")
    # model.load_state_dict(torch.load("./parameter/ResNet/train_100_0.9126999974250793.pth"))

    print("预训练模型加载完成")

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
    """
    测试模型正确率
    :param model: 识别模型
    :param dataloader: 数据加载器
    :param max_counter: 最大测试次数
    :return:
    """
    # 计数器
    counter = 0
    max_counter = min(args.count, len(dataloader))
    print(f"Total Test Num: {max_counter}")
    batch_size = dataloader.batch_size
    # 整体正确率
    total_num = 0
    total_accuracy = 0
    model.eval()

    tqdm_dataloader = tqdm(dataloader, desc="Test:")
    for image, target in tqdm_dataloader:
        image, target = image.to(device), target.to(device)

        # 生成攻击图像
        pert_image = attacker.attack(image, target)
        # 正确模型图像
        output = attacker.forward(pert_image)

        counter += 1
        total_num += batch_size
        accuracy = (output.argmax(1) == target).sum()
        total_accuracy += accuracy

        tqdm_dataloader.set_postfix(No=counter, Acc=f"{accuracy / batch_size}")

        if counter >= max_counter:
            break

    print(f"{attacker.__class__.__name__}正确率: {total_accuracy / (max_counter * batch_size)}")


if __name__ == "__main__":
    main()
