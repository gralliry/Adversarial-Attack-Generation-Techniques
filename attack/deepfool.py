# -*- coding: utf-8 -*-
# @Description:
import torch

import numpy as np

from .base_model import BaseModel


class DeepFool(BaseModel):
    def __init__(self, model, overshoot=0.02, iters=50, cuda=True):
        """
        DeepFool

        https://arxiv.org/abs/1511.04599
        https://github.com/LTS4/DeepFool
        https://github.com/aminul-huq/DeepFool
        https://medium.com/@aminul.huq11/pytorch-implementation-of-deepfool-53e889486ed4
        :param model: 模型
        :param overshoot: 最大限制
        :param iters: 迭代次数
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)

        self.overshoot = overshoot
        self.iters = iters

    def attack(self, image, target):
        """
        # DeepFool
        只接受 batch_size = 1 的数据
        :param image: 需要处理的张量
        :param target: the target is useless, leave it alone
        :return: 生成的对抗样本
        """
        assert image.size(0) == 1, ValueError("只接受 batch_size = 1 的数据")
        image = image.clone().detach()
        # pert_image = self.totensor(image)
        x = image.clone().detach().requires_grad_(True)
        # 获取正确的预测输出
        f_image = self.model(image)
        fs = self.model(x)
        # 获取类别数/标签数
        num_classes = f_image.size(1)
        # 获取所有标签正确率的索引排序
        i_classes = f_image.data[0].argsort().cpu().numpy()[::-1]
        i_classes = i_classes[0:num_classes]
        # 当前模型输出的预测标签
        label = i_classes[0]
        k_i = label
        # 当前标签到正确标签的梯度差距
        w = np.zeros(image.shape)
        r_tot = np.zeros(image.shape)

        self.model.eval()
        with torch.set_grad_enabled(True):
            for _ in range(self.iters):
                # 如果预测后标签仍是正确的标签
                if k_i != label:
                    # 成功生成对抗样本，直接退出
                    break
                # 设定损失最小为0
                pert = 0
                # 对正确标签反向传播
                fs[0, i_classes[0]].backward(retain_graph=True)
                # 当前正确标签的梯度
                grad_orig = x.grad.data.cpu().numpy().copy()

                for k in range(1, num_classes):
                    # 清空梯度 # 防止其他标签的梯度影响当前标签梯度的计算
                    if x.grad is not None:
                        x.grad.zero_()
                    # 对每个对应标签反向传播
                    fs[0, i_classes[k]].backward(retain_graph=True)
                    # 当前标签的梯度
                    cur_grad = x.grad.data.cpu().numpy().copy()
                    # 计算当前标签到正确标签的梯度差距
                    w_k = cur_grad - grad_orig
                    # 计算当前标签输出和正确标签输出之间的差异
                    f_k = (fs[0, i_classes[k]] - fs[0, i_classes[0]]).data.cpu().numpy()
                    # 计算归一化的扰动大小
                    pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
                    # 选择较大的损失并替换
                    if pert_k > pert:
                        pert = pert_k
                        w = w_k

                # 防止 pert 为 0 # 将扰动的大小进行归一化，并维持朝梯度w方向
                r_i = (pert + 1e-4) * w / np.linalg.norm(w)
                r_tot = np.float32(r_tot + r_i)

                # 添加扰动到原图像 生成 对抗样本
                x = image + (1 + self.overshoot) * torch.from_numpy(r_tot).to(self.device)
                x.requires_grad = True

                # 获取添加扰动后图像的标签
                fs = self.model(x)
                k_i = np.argmax(fs.data.cpu().numpy().flatten())

        return x
