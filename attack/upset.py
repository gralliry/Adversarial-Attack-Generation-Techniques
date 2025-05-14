# -*- coding: utf-8 -*-
# @Description:
import torch
from torch import nn

from .base import BaseModel


class ResidualModel(nn.Module):
    """
    UPSET

    For a label, an image is output, a perturbation is generated, and a clean sample + perturbation = an attack sample
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # (batch_size, 3, 32, 32) -> (batch_size, 3, 32, 32)
        return x + self.model1(x)


class UPSET(BaseModel):
    def __init__(self, model, residual_model, cuda=True, s=0.1):
        """
        UPSET
        https://arxiv.org/abs/1707.01159
        :param model: Perturbation generation model. Perturbation generating model, not an identification model
        :param cuda:  Whether to use CUDA
        """
        super().__init__(model=model, cuda=cuda)
        self.s = s
        self.residual_model = residual_model

    def attack(self, image, target, is_targeted=False):
        """
        :param image:
        :param target: useless, leave it alone
        :param is_targeted: useless, leave it alone
        """
        image = image.clone().detach().requires_grad_(True)
        # Superimpose perturbations to the original sample
        # Output perturbations
        pert_image = image + self.s * self.residual_model(image)
        # Limitations
        pert_image = torch.clamp(pert_image, 0, 1)
        return pert_image.detach()
