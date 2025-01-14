# -*- coding: utf-8 -*-
# @Description:
import torch

from .base_model import BaseModel


class MI_FGSM(BaseModel):
    def __init__(self, model, criterion, epsilon=0.1, decay_factor=0.5, iters=10, cuda=True):
        """
        MI_FGSM

        https://arxiv.org/abs/1710.06081

        https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/master/attack/mifgsm.py
        :param model:
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

    def attack(self, image, target):
        """
        MI-FGSM
        :param image:
        :param target: 正确标签
        :return:       生成的对抗样本
        """
        pert_image = image.clone().detach().requires_grad_(True)
        # Generate spoofed labels
        # For example, the target index is generated here, which is the label index of plane
        # attack_target = [(i + 1) % 10 for i in target]

        alpha = self.epsilon / self.iters
        self.model.eval()
        with torch.set_grad_enabled(True):
            for _ in range(self.iters):
                # Set up the gradient
                pert_image.requires_grad = True
                # Forward propagation
                output = self.model(pert_image)
                self.model.zero_grad()
                # Calculate the loss
                loss = self.criterion(output, target)
                loss.backward()
                # Generate adversarial perturbations # Use momentum to update perturbations # Gradient normalization
                grad = pert_image.grad
                grad = self.decay_factor * grad + grad / torch.norm(grad, p=1)
                pert_image = pert_image + alpha * torch.sign(grad)
                # Make sure that the perturbed image is still a valid input (within the range of [0, 1])
                pert_image = torch.clamp(pert_image, 0, 1).detach()
                # When the maximum perturbation is reached, exit directly
                if torch.norm((pert_image - image), p=float('inf')) > self.epsilon:
                    break

        return pert_image
