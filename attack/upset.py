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
    def __init__(self, model: ResidualModel, cuda=True):
        """
        UPSET
        https://arxiv.org/abs/1707.01159
        :param model: Perturbation generation model! Attention: Perturbation generating model, not an identification model
        :param cuda:  Whether to use CUDA
        """
        super().__init__(model=model, cuda=cuda)

    def attack(self, image, target, is_targeted=False):
        """
        :param image:
        :param target: useless, leave it alone
        :param is_targeted: useless, leave it alone
        """
        pert_image = image.clone().detach().requires_grad_(True)
        # Superimpose perturbations to the original sample
        # Output perturbations
        pert_image = pert_image + self.model(pert_image)
        # Limitations
        pert_image = torch.clamp(pert_image, 0, 1)
        return pert_image
