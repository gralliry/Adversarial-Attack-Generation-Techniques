# -*- coding: utf-8 -*-
# @Description:

import torch

from .base import BaseModel


class I_FGSM(BaseModel):
    def __init__(self, model, epsilon=0.2, alpha=0.01, iters=15, cuda=True):
        """
        I-FGSM

        https://github.com/1Konny/FGSM?tab=readme-ov-file
        :param model:
        :param epsilon:   Disturbance amplitude (maximum disturbance limit)
        :param iters:     The number of iterations
        :param cuda:      Whether to start CUDA
        """
        super().__init__(model=model, cuda=cuda)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters

    def attack(self, image, target, is_targeted=False):
        """
        I-FGSM
        :param image:  Tensors that need to be processed
        :param target: Correct tag value
        :param is_targeted: Correct tag value
        :return:       Adversarial sample generated
        """
        pert_image = image.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            # Iterate
            for _ in range(self.iters):
                pert_image = pert_image.clone().detach().requires_grad_(True)
                # Forward propagation
                output = self.model(pert_image)
                self.model.zero_grad()
                # Calculate the loss
                if is_targeted:
                    loss = -self.criterion(output, target)
                else:
                    loss = self.criterion(output, target)
                loss.backward()
                # Gradient Rise # Utilize gradient symbols to perturb while limiting the size of the perturbation
                delta = torch.clamp(self.alpha * pert_image.grad.sign(), min=-self.epsilon, max=self.epsilon)
                # Make sure the perturbed image is still a valid input (in the range of [0, 1])
                pert_image = torch.clamp(pert_image + delta, 0, 1)

        return pert_image.detach()
