# -*- coding: utf-8 -*-
# @Description:
import torch

from .base import BaseModel


class CW(BaseModel):
    def __init__(self, model, cuda=True, c=1, kappa=1, steps=50, lr=0.1):
        """
        C&W attack

        https://arxiv.org/abs/1709.03842

        https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py

        https://colab.research.google.com/drive/1Lc36RwSqvbLTxY6G6O1hkuBn9W49x0jO?usp=sharing#scrollTo=d_a5K75-ZW00
        :param model:
        :param c:
        :param kappa:
        :param lr:
        :param cuda:      Whether to start CUDA
        """
        super().__init__(model=model, cuda=cuda)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def attack(self, image, target, is_targeted=True):
        assert image.size(0) == 1, ValueError("Only data with batch_size = 1 will be accepted")

        image = image.clone().detach().requires_grad_(True)
        # 0, 1 -> -1, 1 -> -inf, inf
        w = self.atanh(torch.clamp(image * 2 - 1, min=-1, max=1)).detach().requires_grad_(True)

        best_adv_images = image.clone().detach()
        best_l2 = 1e10 * torch.ones((len(image),)).to(self.device)
        dim = len(image.shape)

        mse_loss = torch.nn.MSELoss(reduction="none")

        optimizer = torch.optim.Adam([w], lr=self.lr)

        # Start iterating
        with torch.enable_grad():
            for step in range(self.steps):
                # Get adversarial images
                adv_images = self.tanh_space(w)

                # Calculate loss
                current_l2 = mse_loss(adv_images, image).sum(dim=[1, 2, 3])
                l2_loss = current_l2.sum()

                outputs = self.model(adv_images)

                one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[target]
                # (0,0,...,1,...,0)
                # find the max logit other than the target class
                other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
                # get the target class's logit
                real = torch.max(one_hot_labels * outputs, dim=1)[0]

                if is_targeted:
                    f_loss = torch.clamp((other - real), min=-self.kappa)
                else:
                    f_loss = torch.clamp((real - other), min=-self.kappa)

                cost = l2_loss + self.c * f_loss

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                # Update adversarial images
                pre = torch.argmax(outputs.detach(), 1)
                if is_targeted:
                    # We want to let pre == target_labels in a targeted attack
                    condition = (pre == target).float()
                else:
                    # If the attack is not targeted we simply make these two values unequal
                    condition = (pre != target).float()

                # Filter out images that get either correct predictions or non-decreasing loss,
                # i.e., only images that are both misclassified and loss-decreasing are left
                mask = condition * (best_l2 > current_l2.detach())
                best_l2 = mask * current_l2.detach() + (1 - mask) * best_l2

                mask = mask.view([-1] + [1] * (dim - 1))
                best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

        return best_adv_images.detach()

    @staticmethod
    def tanh_space(x):
        # -inf, inf -> 0, 1
        return 1 / 2 * (torch.tanh(x) + 1)

    @staticmethod
    def atanh(x):
        # -1, 1 -> -inf, inf
        return 0.5 * torch.log((1 + x) / (1 - x))
