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

if __name__ == "__main__":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_datasets = datasets.CIFAR10("./datasets", train=False, transform=transform_test)

    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    # 指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    model = ResNet18().to(device)
    model.load_state_dict(torch.load("./model/ResNet/train_100_0.9126999974250793.pth"))

    # model = models.vgg19().to(device)
    # model.load_state_dict(torch.load("./model/pretrained/checkpoints/vgg19-dcbb9e9d.pth"))

    # torch.hub.set_dir("./model/pretrained")
    print("预训练模型加载完成")

    # TEST
    # base_model = BaseModel(model=model)
    # base_model.test_attack(model=model, dataloader=test_dataloader)

    # L-BFGS
    # l_bfgs_attacker = L_BFGS(model=model, criterion=criterion, iters=2, epsilon=0.2)
    # l_bfgs_attacker.test_attack(model=model, dataloader=test_dataloader)

    # FGSM
    # fgsm_attacker = FGSM(model=model, criterion=criterion, epsilon=0.2)
    # fgsm_attacker.test_attack(model=model, dataloader=test_dataloader)

    # I_FGSM
    # i_fgsm_attacker = I_FGSM(model=model, criterion=criterion)
    # i_fgsm_attacker.test_attack(model=model, dataloader=test_dataloader)

    # MI_FGSM
    # mi_fgsm_attacker = MI_FGSM(model=model)
    # mi_fgsm_attacker.test_attack(model=model, dataloader=test_dataloader)

    # DeepFool
    # deepfool_attacker = DeepFool(model=model)
    # deepfool_attacker.test_attack(model=model, dataloader=test_dataloader, max_counter=3000)

    # JSMA
    jsma_attacker = JSMA(model=model)
    jsma_attacker.test_attack(model=model, dataloader=test_dataloader, max_counter=500)

    # ONE-PIXEL
    # one_pixel_attacker = ONE_PIXEL(model=model, iters=10)
    # one_pixel_attacker.test_attack(model=model, dataloader=test_dataloader, max_counter=500)

    # C&W attack
    # cw_attacker = CW(model=model, iters=1000)
    # cw_attacker.test_attack(model=model, dataloader=test_dataloader)

    # UPSET
    # residual_model = ResidualModel().to(device)
    # residual_model.load_state_dict(torch.load("./model/UPSET/target_0/0.9653946161270142.pth"))
    # upset_attacker = UPSET(model=residual_model)
    # upset_attacker.test_attack(model=model, dataloader=test_dataloader)

    ...

    ...
