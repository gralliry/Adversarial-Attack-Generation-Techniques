# -*- coding: utf-8 -*-
# @Description:
import torch

from .base import BaseModel


class CW(BaseModel):
    def __init__(self, model, c=1, alpha=1, iters=20, cuda=True):
        """
        C&W attack

        https://arxiv.org/abs/1709.03842

        https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py

        https://colab.research.google.com/drive/1Lc36RwSqvbLTxY6G6O1hkuBn9W49x0jO?usp=sharing#scrollTo=d_a5K75-ZW00
        :param model:
        :param c:         Perturbation step size
        :param alpha:        The probability of retaining the perturbation point
        :param iters:     The number of iterations
        :param cuda:      Whether to start CUDA
        """
        super().__init__(model=model, cuda=cuda)
        self.c = c
        self.alpha = alpha
        self.iters = iters

    def attack(self, image, target, is_targeted=True):
        # is_targeted is useless
        assert image.size(0) == 1, ValueError("Only data with batch_size = 1 will be accepted")

        image = image.clone().detach().requires_grad_(True)
        pert_image = image.clone().detach().requires_grad_(True)

        # Start iterating
        with torch.set_grad_enabled(True):
            for _ in range(self.iters):
                self.model.zero_grad()
                output = self.model(pert_image)
                # The attack tag has been reached
                if is_targeted:
                    if output.argmax(1) == target:
                        break
                else:
                    if output.argmax(1) != target:
                        break
                loss = torch.norm(pert_image - image, p=2)
                loss += self.c * torch.nn.functional.nll_loss(output, target)
                loss.backward()  # 反向传播
                # Overlay Perturbation # Clips the perturbated image to the range of [0,1].
                pert_image = torch.clamp(pert_image + self.alpha * pert_image.grad, 0, 1).requires_grad_(True)
                # It is possible that
                # the number of iterations reaches the upper limit and none of the specified attack tags are reached
                # If so, it may need to be seen as a failed attack

        return pert_image
