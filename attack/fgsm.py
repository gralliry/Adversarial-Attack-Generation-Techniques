# -*- coding: utf-8 -*-
# @Description:

import torch

from .base import BaseModel


class FGSM(BaseModel):
    def __init__(self, model, epsilon=0.06, cuda=True):
        """
        FGSM

        https://github.com/1Konny/FGSM?tab=readme-ov-file

        https://github.com/Harry24k/FGSM-pytorch/blob/master/FGSM.ipynb
        :param model:
        :param epsilon:   Amplitude of disturbance
        """
        super().__init__(model=model, cuda=cuda)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.epsilon = epsilon

    def attack(self, image, target, is_targeted=False):
        """
        FGSM
        :param image:  Tensors that need to be processed
        :param target: tag value
        :param is_targeted:
        :return:       Adversarial sample generated
        """
        # Set the requires_grad of the input tensor to True to calculate the gradient
        pert_image = image.clone().detach().requires_grad_(True)
        # The evaluation mode is set, but the gradient is calculated normally
        # self.model.eval()
        with torch.enable_grad():
            # Use the model for forward propagation
            output = self.model(pert_image)
            # Calculate the loss function
            if is_targeted:
                loss = -self.criterion(output, target)
            else:
                loss = self.criterion(output, target)
            # Backpropagation, calculating the gradient
            loss.backward()
            # Limit the generated adversarial samples to the range of [0, 1].
            pert_image = torch.clamp(pert_image + self.epsilon * pert_image.grad.sign(), 0, 1)

        return pert_image.detach()
