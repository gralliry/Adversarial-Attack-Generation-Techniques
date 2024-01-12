# -*- coding: utf-8 -*-
# @Time    : 2024/1/7 18:29
# @Author  : Liang Jinaye
# @File    : fgsm.py
# @Description :

import torch

from .model import BaseModel


class FGSM(BaseModel):
    def __init__(self, model, criterion, epsilon=0.06, cuda=True):
        """
        FGSM
        :param model: 模型
        :param criterion: 损失函数
        :param epsilon: 扰动幅度
        """
        super().__init__(model=model, cuda=cuda)

        self.criterion = criterion.to(self.device)
        self.epsilon = epsilon

    def attack(self, image, target):
        """
        FGSM
        :param image: 需要处理的张量
        :param target: 正确的标签值
        :return: 生成的对抗样本
        """
        # https://github.com/1Konny/FGSM?tab=readme-ov-file
        # https://github.com/Harry24k/FGSM-pytorch/blob/master/FGSM.ipynb
        # 设置输入张量的 requires_grad 为 True 计算梯度
        pert_image = self.totensor(image, requires_grad=True)
        target = self.totensor(target)
        # 设置评估模式，但正常计算梯度
        self.model.eval()
        with torch.set_grad_enabled(True):
            # 使用模型进行前向传播
            output = self.model(pert_image)
            # 将模型参数的梯度归零
            self.model.zero_grad()
            # 计算损失函数
            loss = self.criterion(output, target)
            # 反向传播，计算梯度
            loss.backward()
            # 进行梯度上升 # 利用梯度符号进行扰动
            pert_image = pert_image + self.epsilon * pert_image.grad.sign()
            # 将生成的对抗样本限制在[0, 1]范围内
            pert_image = torch.clamp(pert_image, 0, 1)

        return pert_image
