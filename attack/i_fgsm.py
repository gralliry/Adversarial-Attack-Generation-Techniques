# -*- coding: utf-8 -*-
# @Description:

import torch

from .base_model import BaseModel


class I_FGSM(BaseModel):
    def __init__(self, model, criterion, epsilon=0.2, iters=15, cuda=True):
        """
        I-FGSM

        https://github.com/1Konny/FGSM?tab=readme-ov-file
        :param model:
        :param criterion: Loss function
        :param epsilon:   Disturbance amplitude (maximum disturbance limit)
        :param iters:     The number of iterations
        :param cuda:      Whether to start CUDA
        """
        super().__init__(model=model, cuda=cuda)

        self.criterion = criterion.to(self.device)
        self.epsilon = epsilon
        self.iters = iters

    def attack(self, image, target):
        """
        I-FGSM
        :param image:  Tensors that need to be processed
        :param target: Correct tag value
        :return:       Adversarial sample generated
        """
        pert_image = image.clone().detach().requires_grad_(True)
        # Iteration step size
        alpha = self.epsilon / self.iters

        self.model.eval()
        with torch.set_grad_enabled(True):
            # Iterate
            for _ in range(self.iters):
                # Forward propagation
                outputs = self.model(pert_image)
                self.model.zero_grad()
                # Calculate the loss
                loss = self.criterion(outputs, target)
                loss.backward()
                # Gradient Rise # Utilize gradient symbols to perturb while limiting the size of the perturbation
                pert_image = pert_image + alpha * pert_image.grad.sign()
                # Make sure the perturbed image is still a valid input (in the range of [0, 1])
                pert_image = torch.clamp(pert_image, 0, 1)
                # When the maximum perturbation is reached, exit directly
                if torch.norm((pert_image - image), p=float('inf')) > self.epsilon:
                    break

        return pert_image
