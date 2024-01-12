# -*- coding: utf-8 -*-
# @Time    : 2024/1/7 18:48
# @Author  : Liang Jinaye
# @File    : deepfool.py
# @Description :
import torch

import numpy as np

from .model import BaseModel


class DeepFool(BaseModel):
    def __init__(self, model, overshoot=0.02, max_iter=50, cuda=True):
        """
        DeepFool
        :param model: 模型
        :param overshoot:
        :param max_iter: 最大迭代次数
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)

        self.overshoot = overshoot
        self.max_iter = max_iter

    def test_attack_args(self, image, **kwargs):
        return (image,)

    def attack(self, image):
        """
        # DeepFool
        只接受 batch_size = 1 的数据
        https://medium.com/@aminul.huq11/pytorch-implementation-of-deepfool-53e889486ed4
        https://github.com/aminul-huq/DeepFool
        :param image: 需要处理的张量
        :return: 生成的对抗样本
        """
        assert image.size(0) == 1, ValueError("只接受 batch_size = 1 的数据")

        image = self.totensor(image)

        f_image = self.model(image)
        # 获取类别数
        num_classes = f_image.size(1)

        f_image = f_image.data.cpu().numpy().flatten()
        i_classes = (np.array(f_image)).flatten().argsort()[::-1]

        i_classes = i_classes[0:num_classes]
        label = i_classes[0]

        pert_image = self.totensor(image)

        input_shape = image.shape
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0

        # x = Variable(pert_image[None, :], requires_grad=True)
        # x = torch.tensor(pert_image, requires_grad=True)
        x = self.totensor(pert_image, requires_grad=True)
        fs = self.model.forward(x)

        # 从这里开始循环
        # fs_list = [fs[0, i_classes[k]] for k in range(num_classes)]
        k_i = label

        while k_i == label and loop_i < self.max_iter:

            pert = np.inf
            fs[0, i_classes[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, num_classes):
                if x.grad is not None:
                    x.grad.zero_()

                fs[0, i_classes[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, i_classes[k]] - fs[0, i_classes[0]]).data.cpu().numpy()

                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            pert_image = image + (1 + self.overshoot) * torch.from_numpy(r_tot).to(self.device)
            # x = Variable(pert_image, requires_grad=True)
            x = pert_image.clone().detach().requires_grad_(True)
            fs = self.model(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1
        # r_tot = (1 + self.overshoot) * r_tot
        # return r_tot, loop_i, label, k_i, pert_image
        return pert_image
