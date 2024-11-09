# -*- coding: utf-8 -*-
# @Description:
import torch

from .base_model import BaseModel


class CW(BaseModel):
    def __init__(self, model, criterion, a=1, cr=1, iters=20, cuda=True):
        """
        C&W attack

        https://arxiv.org/abs/1709.03842

        https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py

        https://colab.research.google.com/drive/1Lc36RwSqvbLTxY6G6O1hkuBn9W49x0jO?usp=sharing#scrollTo=d_a5K75-ZW00
        :param model: 模型
        :param criterion: 损失函数
        :param a: 扰动步长
        :param cr: 保留扰动点的概率
        :param iters: 迭代次数
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)
        self.criterion = criterion
        self.a = a
        self.cr = cr
        self.iters = iters

    def attack(self, image, target):
        assert image.size(0) == 1, ValueError("只接受 batch_size = 1 的数据")

        image = image.clone().detach().requires_grad_(True)
        pert_image = image.clone().detach().requires_grad_(True)
        attack_target = (target + 1) % 10

        output = self.model(pert_image)
        self.model.zero_grad()

        loss = self.criterion(output, attack_target)
        loss.backward()

        # 获取初始梯度
        grad = pert_image.grad.data
        total_grad = torch.zeros_like(grad)

        pert_image = pert_image - self.a * grad

        # 开始迭代
        with torch.set_grad_enabled(True):
            for _ in range(self.iters):
                # 梯度调整 # 选择梯度上升或下降 # 获取新的梯度和损失
                output, grad, loss = self.gradient_adjust(pert_image, loss, attack_target)
                # 已到达攻击标签
                if output.argmax(1) == attack_target:
                    break
                # 累加梯度
                total_grad += grad
                # 叠加扰动 # 将扰动图像剪切到[0,1]范围内
                pert_image = torch.clamp(pert_image + self.a * grad, 0, 1).requires_grad_(True)
                # 有可能迭代次数达到上限任未到达指定的攻击标签

        # 计算平均累加的梯度
        r = (self.a / self.iters) * total_grad
        # r = pert_image - image
        # 二分优化
        pert_image = self.binary_optimize(output, image, image + r)
        r = pert_image - image
        # 以一定概率保留部分点
        mask = (torch.rand(image.shape) < self.cr).to(self.device)
        pert_image = torch.clamp(image + r * mask, 0, 1)

        return pert_image

    def gradient_adjust(self, new_image, loss, attack_target):
        """
        梯度调整
        :param new_image: X(t)
        :param loss: Loss(t-1)
        :param attack_target: 攻击的标签
        :return:
        """
        # 获取新样本的输出
        new_image = new_image.clone().detach().requires_grad_(True)
        new_output = self.model(new_image)
        self.model.zero_grad()
        new_loss = self.criterion(new_output, attack_target)
        new_loss.backward(retain_graph=True)
        # 获取新样本的损失和梯度
        if new_loss < loss:
            # 梯度下降
            return new_output, -new_image.grad.data.clone(), new_loss
        else:
            # 梯度上升
            return new_output, new_image.grad.data.clone(), new_loss

    def binary_optimize(self, output, l_image, r_image):
        """
        二分优化
        :param output: 对抗样本的输出
        :param l_image: 原样本
        :param r_image: 对抗样本
        :return:
        """
        # 定义相对误差和绝对误差的阈值
        rtol = 0.01
        atol = 0.01
        # 小于一定误差结束循环
        while not torch.isclose(l_image, r_image, rtol=rtol, atol=atol).all():
            m_image = (l_image + r_image) / 2
            m_output = self.model(m_image)
            # 当 sub_l_image 和 sub_r_image 接近相等，直接返回
            if m_output.argmax(1) != output.argmax(1):
                # 如果与正确标签不相同，则将扰动减少些
                r_image = m_image
            else:
                # 如果与正确标签相同，则将扰动扩大些
                l_image = m_image

        return 2 * r_image - l_image
