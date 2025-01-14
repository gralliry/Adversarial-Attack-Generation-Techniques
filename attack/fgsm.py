# -*- coding: utf-8 -*-
# @Description:

import torch

from .base_model import BaseModel


class FGSM(BaseModel):
    def __init__(self, model, criterion, epsilon=0.06, cuda=True):
        """
        FGSM

        https://github.com/1Konny/FGSM?tab=readme-ov-file

        https://github.com/Harry24k/FGSM-pytorch/blob/master/FGSM.ipynb
        :param model:
        :param criterion: Loss function
        :param epsilon:   Amplitude of disturbance
        """
        super().__init__(model=model, cuda=cuda)

        self.criterion = criterion.to(self.device)
        self.epsilon = epsilon

    def attack(self, image, target):
        """
        FGSM
        :param image:  Tensors that need to be processed
        :param target: Correct tag value
        :return:       Adversarial sample generated
        """
        # Set the requires_grad of the input tensor to True to calculate the gradient
        pert_image = image.clone().detach().requires_grad_(True)
        # The evaluation mode is set, but the gradient is calculated normally
        self.model.eval()
        with torch.set_grad_enabled(True):
            # Use the model for forward propagation
            output = self.model(pert_image)
            # Zeroing the gradient of the model parameters
            self.model.zero_grad()
            # Calculate the loss function
            loss = self.criterion(output, target)
            # Backpropagation, calculating the gradient
            loss.backward()
            # Perform a gradient ascent # Perturbation with gradient symbols
            pert_image = pert_image + self.epsilon * pert_image.grad.sign()
            # Limit the generated adversarial samples to the range of [0, 1].
            pert_image = torch.clamp(pert_image, 0, 1)

        return pert_image
