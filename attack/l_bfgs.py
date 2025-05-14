# -*- coding: utf-8 -*-
# @Description:
import torch
import torch.optim as optim

from .base import BaseModel


class L_BFGS(BaseModel):
    def __init__(self, model, epsilon=0.1, alpha=0.1, iters=10, lr=0.01, cuda=True):
        """
        L_BFGS
        :param model: 攻击的模型
        :param epsilon: 扰动大小
        :param alpha: 权重
        :param iters: 迭代次数
        :param lr: 学习率
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.lr = lr

    def attack(self, image, target, is_targeted=True):
        """
        :param image: 需要攻击的样本
        :param target: 攻击的标签，这里可替换成需要攻击的标签
        :param is_targeted:
        :return: 对抗样本
        """
        pert_image = image.clone().detach().to(self.device)
        # 生成扰动 # 正则化项，旨在控制扰动的大小，防止扰动过大
        r = (self.epsilon * torch.rand(image.shape)).to(self.device).requires_grad_(True)
        # 优化器 # 创建一个Adam优化器，用于更新扰动 r 的数值
        optimizer = optim.Adam([r], lr=self.lr)
        with torch.enable_grad():
            for _ in range(self.iters):
                # 干净样本 + 扰动 = 对抗样本
                pert_image = torch.clamp(pert_image + r, 0, 1).detach().requires_grad_(True)
                output = self.model(pert_image)
                # 模型输出与目标标签之间的损失，即希望模型在对抗样本上产生与 attack_target 相关的错误预测
                if is_targeted:
                    loss = self.alpha * r.abs().sum() + self.criterion(output, target)
                else:
                    loss = self.alpha * r.abs().sum() - self.criterion(output, target)
                # 反向传播，修正参数
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return pert_image.detach()
