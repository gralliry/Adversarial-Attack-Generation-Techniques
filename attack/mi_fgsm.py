# -*- coding: utf-8 -*-
# @Description:
import torch

from .model import BaseModel


class MI_FGSM(BaseModel):
    def __init__(self, model, criterion, epsilon=0.1, decay_factor=0.5, iters=10, cuda=True):
        """
        MI_FGSM

        https://arxiv.org/abs/1710.06081

        https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/master/attack/mifgsm.py
        :param model: 模型
        :param criterion: 损失函数
        :param epsilon: 扰动
        :param decay_factor: 衰减因子
        :param iters: 迭代次数
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)

        self.criterion = criterion.to(self.device)
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.iters = iters

    def test_attack_args(self, image, target, **kwargs):
        # 生成欺骗标签
        # 比如这里生成目标索引为0，即plane的标签索引
        # attack_target = [(i + 1) % 10 for i in target]
        # return image, attack_target
        return image, target

    def attack(self, image, target):
        """
        MI-FGSM
        :param image: 需要处理的张量
        :param target: 正确标签
        :return: 生成的对抗样本
        """
        pert_image = image.clone().detach().requires_grad_(True)
        target = self.totensor(target)

        alpha = self.epsilon / self.iters
        self.model.eval()
        with torch.set_grad_enabled(True):
            for _ in range(self.iters):
                # 设置梯度
                pert_image.requires_grad = True
                # 前向传播
                output = self.model(pert_image)
                self.model.zero_grad()
                # 计算损失（针对目标攻击的负对数似然）
                loss = self.criterion(output, target)
                loss.backward()
                # 生成对抗扰动 # 使用动量来更新扰动 # 梯度归一化
                grad = pert_image.grad.sign()
                grad = self.decay_factor * grad + grad / torch.norm(grad, p=1)
                pert_image = pert_image + alpha * torch.sign(grad)
                # 确保扰动后的图像仍然是有效的输入（在 [0, 1] 范围内）
                pert_image = torch.clamp(pert_image, 0, 1).detach()
                # 达到最大扰动，直接退出
                if torch.norm((pert_image - image), p=float('inf')) > self.epsilon:
                    break

        return pert_image
