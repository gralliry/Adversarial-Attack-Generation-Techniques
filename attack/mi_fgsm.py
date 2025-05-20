# -*- coding: utf-8 -*-
# @Description:
import torch

from .base import BaseModel


class MI_FGSM(BaseModel):
    def __init__(self, model, epsilon=0.03, alpha=0.01, decay=0.5, iters=10, cuda=True):
        """
        MI_FGSM

        https://arxiv.org/abs/1710.06081

        https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/master/attack/mifgsm.py
        :param model:
        :param decay: 衰减因子
        :param iters: 迭代次数
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.decay = decay
        self.iters = iters

    def attack(self, image, target, is_targeted=False):
        """
        MI-FGSM
        :param image:
        :param target: 正确标签
        :param is_targeted:
        :return:       生成的对抗样本
        """
        image = image.clone().detach().requires_grad_(True)
        pert_image = image.clone().detach().requires_grad_(True)
        # Generate spoofed labels
        momentum = torch.zeros_like(pert_image).requires_grad_(True)
        with torch.enable_grad():
            for _ in range(self.iters):
                # Forward propagation
                output = self.model(pert_image)
                # Calculate the loss
                if is_targeted:
                    loss = -self.criterion(output, target)
                else:
                    loss = self.criterion(output, target)
                loss.backward()
                # Generate adversarial perturbations # Use momentum to update perturbations # Gradient normalization
                grad = pert_image.grad.detach()
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * self.decay
                momentum = grad
                # Make sure that the perturbed image is still a valid input (within the range of [0, 1])
                pert_image = pert_image.detach() + self.alpha * grad.sign()
                delta = torch.clamp(pert_image - image, min=-self.epsilon, max=self.epsilon)
                pert_image = torch.clamp(image + delta, min=0, max=1).detach().requires_grad_(True)

        return pert_image.detach()
