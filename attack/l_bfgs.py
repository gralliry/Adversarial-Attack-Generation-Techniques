# -*- coding: utf-8 -*-
# @Description:
import torch
import torch.optim as optim

from .base_model import BaseModel


class L_BFGS(BaseModel):
    def __init__(self, model, criterion, epsilon=0.1, alpha=0.1, iters=1, lr=0.001, cuda=True):
        """
        L_BFGS
        :param model: 攻击的模型
        :param criterion: 损失函数
        :param epsilon: 扰动大小
        :param alpha: 权重
        :param iters: 迭代次数
        :param lr: 学习率
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)

        self.criterion = criterion
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.lr = lr

    def attack(self, image, target):
        """
        :param image: 需要攻击的样本
        :param attack_target: 攻击的标签
        :return: 对抗样本
        """
        pert_image = image.clone().detach().to(self.device)
        # 生成错误标签，这里可替换成需要攻击的标签
        attack_target = (target + 1) % 10

        self.model.eval()

        # 生成扰动 # 正则化项，旨在控制扰动的大小，防止扰动过大
        r = (self.epsilon * torch.rand(pert_image.shape)).to(self.device).requires_grad_(True)
        # 优化器 # 创建一个Adam优化器，用于更新扰动 r 的数值
        optimizer = optim.Adam([r], lr=self.lr)
        for _ in range(self.iters):
            # 干净样本 + 扰动 = 对抗样本
            pert_image = torch.clamp(pert_image + r, 0, 1).detach()
            outputs = self.model(pert_image)
            # 模型输出与目标标签之间的损失，即希望模型在对抗样本上产生与 attack_target 相关的错误预测
            loss = self.alpha * r.abs().sum() + self.criterion(outputs, attack_target)

            # 反向传播，修正参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return pert_image
