# -*- coding: utf-8 -*-
# @Description:
import torch
import numpy as np

from .base_model import BaseModel


class JSMA(BaseModel):
    def __init__(self, model, alpha=3.0, gamma=3.0, iters=20, cuda=True):
        """
        JSMA

        https://arxiv.org/abs/1511.07528

        https://github.com/probabilistic-jsmas/probabilistic-jsmas

        https://github.com/guidao20/MJSMA_JSMA/blob/master/MJSMA_JSMA.py

        https://github.com/FenHua/Adversarial-Examples/blob/master/%E9%BB%91%E7%9B%92/JSMA/JSMA.ipynb
        :param model: 模型
        :param alpha: 扰动步长
        :param gamma: 定义变化极限/边界
        :param iters: 最大循环/寻找次数/改变的像素的数量
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)

        self.alpha = alpha
        self.gamma = gamma
        self.iters = iters

    def attack(self, image, target):
        """
        JSMA
        :param image: 图像
        :param attack_target: 攻击的标签
        :return:
        """
        assert image.size(0) == 1, ValueError("只接受 batch_size = 1 的数据")

        pert_image = image.clone().detach().requires_grad_(True)
        # 生成欺骗标签
        # 这里的fool_target元素数量要和batch_size相同
        # 这里只是单纯生成错误的标签，并没有指定标签
        attack_target = (target + 1) % 10
        # 定义搜索域，修改后的位置置零，下一次不再计算
        mask = np.ones(pert_image.shape)
        # 评估模式
        self.model.eval()
        with torch.set_grad_enabled(True):
            for _ in range(self.iters):
                output = self.model(pert_image)
                # 这里仅适合 batch_size 为 1 的判断
                if output.argmax(1) == attack_target:
                    # 攻击成功，停止迭代
                    break
                # 梯度清零
                if pert_image.grad is not None:
                    pert_image.grad.zero_()
                # 对每个图像进行反向传播
                output[0, attack_target[0]].backward(retain_graph=True)
                # 生成扰动点和扰动大小
                index, pix_sign = self.saliency_map(pert_image, mask)
                # 添加 扰动 到 对抗样本
                pert_image.data[index] += pix_sign * self.alpha * self.gamma
                # 达到极限的点不再参与更新
                if not -self.gamma <= pert_image.data[index] <= self.gamma:
                    # 限制扰动
                    pert_image.data[index] = torch.clamp(pert_image.data[index], -self.gamma, self.gamma)
                    # 搜索域对应的像素置零，表示该点不再参与计算更新
                    mask[index] = 0

        return pert_image

    # 计算显著图
    @staticmethod
    def saliency_map(image, mask):
        """
        此方法为beta参数的简化版本，注重攻击目标贡献大的点
        :param image: 输入的图像
        :param mask: 标记位，记录已经访问的点的坐标
        :return:
        """
        derivative = image.grad.data.cpu().numpy().copy()
        # 预测 对攻击目标的贡献 # 对于搜索过的点设置为0
        alphas = derivative * mask
        # 预测对非攻击目标的贡献
        betas = -np.ones_like(alphas)
        # 计算正向扰动和负向扰动的差距
        sal_map = np.abs(alphas) * np.abs(betas) * np.sign(alphas * betas)
        # 最佳像素和扰动方向 # 有目标攻击选择
        index = np.argmax(sal_map)
        # 转换成(p1,p2)格式
        index = np.unravel_index(index, mask.shape)
        pixel_sign = np.sign(alphas)[index]
        return index, pixel_sign
