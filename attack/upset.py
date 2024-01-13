# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 10:54
# @Author  : Liang Jinaye
# @File    : upset.py
# @Description :
import torch
from torch import nn

from .model import BaseModel


class ResidualModel(nn.Module):
    """
    UPSET

    针对某个标签，输出图像，生成扰动，干净样本 + 扰动 = 攻击样本
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # (batch_size, 3, 32, 32) -> (batch_size, 3, 32, 32)
        x = self.model(x)
        return x


class UPSET(BaseModel):
    def __init__(self, model: ResidualModel, alpha=0.01, iters=5, cuda=True):
        """
        UPSET

        https://arxiv.org/abs/1707.01159
        :param model: 扰动生成模型 ! 扰动生成模型，不是识别模型
        :param alpha: 迭代步长
        :param iters: 迭代次数
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)

        self.alpha = alpha
        self.iters = iters

    def test_attack_args(self, image, target, **kwargs):
        return (image,)

    def attack(self, image):
        pert_image = self.totensor(image)

        for _ in range(self.iters):
            # 输出扰动
            residual = self.model(pert_image)
            # 叠加扰动到原样本
            pert_image = pert_image + self.alpha * residual
            # 限制范围
            pert_image = torch.clamp(pert_image, 0, 1)

        return pert_image
