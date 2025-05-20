# -*- coding: utf-8 -*-
# @Description:
import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseModel


class BasicBlock(nn.Module):
    def __init__(self, planes):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        x = x + self.layer(x)
        return F.relu(x)


class ResidualModel(nn.Module):
    """
    UPSET

    For a label, an image is output, a perturbation is generated, and a clean sample + perturbation = an attack sample
    """

    def __init__(self):
        super(ResidualModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.layer1 = BasicBlock(128)
        self.layer2 = BasicBlock(128)
        self.layer3 = BasicBlock(128)
        self.layer4 = BasicBlock(128)
        self.final = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.final(out)
        # return ((torch.sigmoid(out) - 0.5) * 2
        return out


class UPSET(BaseModel):
    def __init__(self, model, residual_model, cuda=True):
        """
        UPSET
        https://arxiv.org/abs/1707.01159
        :param model: Perturbation generation model. Perturbation generating model, not an identification model
        :param cuda:  Whether to use CUDA
        """
        super().__init__(model=model, cuda=cuda)
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
        pert_image = image + self.residual_model(image)
        # Limitations
        pert_image = torch.clamp(pert_image, 0, 1)
        return pert_image.detach()
