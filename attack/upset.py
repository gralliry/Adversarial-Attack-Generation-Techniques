# -*- coding: utf-8 -*-
# @Description:
import torch
from torch import nn

from .base_model import BaseModel


class ResidualModel(nn.Module):
    """
    UPSET

    For a label, an image is output, a perturbation is generated, and a clean sample + perturbation = an attack sample
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # (batch_size, 3, 32, 32) -> (batch_size, 3, 32, 32)
        x = self.model(x)
        return x


class UPSET(BaseModel):
    def __init__(self, model: ResidualModel, alpha=0.01, iters=5, cuda=True):
        """
        UPSET

        https://arxiv.org/abs/1707.01159
        :param model: Perturbation generation model! Attention: Perturbation generates a model, not an identification model
        :param alpha: Iteration step size
        :param iters: The number of iterations
        :param cuda:  Whether to start CUDA
        """
        super().__init__(model=model, cuda=cuda)

        self.alpha = alpha
        self.iters = iters

    def attack(self, image, target):
        """
        :param image:
        :param target: the target is useless, leave it alone
        """
        pert_image = image.clone().detach().requires_grad_(True)

        for _ in range(self.iters):
            # Output perturbations
            residual = self.model(pert_image)
            # Superimpose perturbations to the original sample
            pert_image = pert_image + self.alpha * residual
            # Limitations
            pert_image = torch.clamp(pert_image, 0, 1)

        return pert_image
