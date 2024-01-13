# -*- coding: utf-8 -*-
# @Time    : 2024/1/7 18:36
# @Author  : Liang Jinaye
# @File    : i_fgsm.py
# @Description :

import torch

from .model import BaseModel


class I_FGSM(BaseModel):
    def __init__(self, model, criterion, epsilon=0.2, iters=15, cuda=True):
        """
        I-FGSM

        https://github.com/1Konny/FGSM?tab=readme-ov-file
        :param model: 模型
        :param criterion: 损失函数
        :param epsilon: 扰动幅度（最大扰动限制）
        :param iters: 迭代次数
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)

        self.criterion = criterion.to(self.device)
        self.epsilon = epsilon
        self.iters = iters

    def attack(self, image, target):
        """
        I-FGSM
        :param image: 需要处理的张量
        :param target: 正确的标签值
        :return: 生成的对抗样本
        """
        pert_image = self.totensor(image, requires_grad=True)
        target = self.totensor(target)
        # 迭代步长
        alpha = self.epsilon / self.iters

        self.model.eval()
        with torch.set_grad_enabled(True):
            # 进行迭代
            for _ in range(self.iters):
                # 正向传播
                outputs = self.model(pert_image)
                self.model.zero_grad()
                # 计算损失（针对目标攻击的负对数似然）
                loss = self.criterion(outputs, target)
                loss.backward()
                # 梯度上升 # 利用梯度符号进行扰动，同时限制扰动的大小
                pert_image = pert_image + alpha * pert_image.grad.sign()
                # 确保扰动后的图像仍然是有效的输入（在 [0, 1] 范围内）
                pert_image = torch.clamp(pert_image, 0, 1)
                # 达到最大扰动，直接退出
                if torch.norm((pert_image - image), p=float('inf')) > self.epsilon:
                    break

        return pert_image
