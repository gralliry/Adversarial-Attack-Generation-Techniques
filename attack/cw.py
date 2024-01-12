# -*- coding: utf-8 -*-
# @Time    : 2024/1/8 20:34
# @Author  : Liang Jinaye
# @File    : cw.py
# @Description :
import torch

from torch import optim

from .model import BaseModel


class CW(BaseModel):
    def __init__(self, model, criterion, c=0.4, kappa=0, learning_rate1=0.03, iters=50, cuda=True):
        """
        https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
        https://colab.research.google.com/drive/1Lc36RwSqvbLTxY6G6O1hkuBn9W49x0jO?usp=sharing#scrollTo=d_a5K75-ZW00
        :param model:
        """
        super().__init__(model=model, cuda=cuda)

        self.criterion = criterion
        self.c = c
        self.kappa = kappa
        self.learning_rate = learning_rate1
        self.iters = iters

    def test_attack_args(self, image, target, **kwargs):
        attack_target = [(i + 1) % 10 for i in target]
        return image, attack_target

    def attack(self, image, attack_target):
        assert image.size(0) == 1, ValueError("只接受 batch_size = 1 的数据")
        self.model.eval()
        image = self.totensor(image)
        attack_target = self.totensor(attack_target)
        epsilon = torch.zeros_like(image, requires_grad=True, device=self.device)
        # 定义优化器
        optimizer = optim.Adam([image], lr=self.learning_rate)
        # 开始迭代
        for _ in range(self.iters):
            output = self.model(image)
            # 如果已经成功生成对抗样本，则退出循环
            if output.argmax(1) == attack_target:
                break
            # 计算损失
            loss = torch.sum(self.c * self.criterion(output, attack_target)) + torch.sum(torch.abs(epsilon))
            # 反向传播以获取梯度
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # 叠加扰动 # 将扰动图像剪切到[0,1]范围内
            image = torch.clamp(image + epsilon, 0, 1)

        return image

    def criterion(self, output, target):
        one_hot_labels = torch.eye(len(output[0]))[target].to(self.device)
        i, _ = torch.max((1 - one_hot_labels) * output, dim=1)
        j = torch.masked_select(output, one_hot_labels.bool())
        return torch.clamp(i - j, min=-self.kappa)
