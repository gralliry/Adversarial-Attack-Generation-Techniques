# -*- coding: utf-8 -*-
# @Description:
import torch

from .base_model import BaseModel


class CW(BaseModel):
    def __init__(self, model, criterion, a=1, cr=1, iters=20, cuda=True):
        """
        C&W attack

        https://arxiv.org/abs/1709.03842

        https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py

        https://colab.research.google.com/drive/1Lc36RwSqvbLTxY6G6O1hkuBn9W49x0jO?usp=sharing#scrollTo=d_a5K75-ZW00
        :param model:
        :param criterion: Loss function
        :param a:         Perturbation step size
        :param cr:        The probability of retaining the perturbation point
        :param iters:     The number of iterations
        :param cuda:      Whether to start CUDA
        """
        super().__init__(model=model, cuda=cuda)
        self.criterion = criterion
        self.a = a
        self.cr = cr
        self.iters = iters

    def attack(self, image, target):
        assert image.size(0) == 1, ValueError("Only data with batch_size = 1 will be accepted")

        image = image.clone().detach().requires_grad_(True)
        pert_image = image.clone().detach().requires_grad_(True)
        attack_target = (target + 1) % 10

        output = self.model(pert_image)
        self.model.zero_grad()

        loss = self.criterion(output, attack_target)
        loss.backward()

        # Get the initial gradient
        grad = pert_image.grad.data
        total_grad = torch.zeros_like(grad)

        pert_image = pert_image - self.a * grad

        # Start iterating
        with torch.set_grad_enabled(True):
            for _ in range(self.iters):
                # Gradient adjustment # Select gradient up or down # Get new gradients and losses
                output, grad, loss = self.gradient_adjust(pert_image, loss, attack_target)
                # The attack tag has been reached
                if output.argmax(1) == attack_target:
                    break
                # Cumulative gradients
                total_grad += grad
                # Overlay Perturbation # Clips the perturbated image to the range of [0,1].
                pert_image = torch.clamp(pert_image + self.a * grad, 0, 1).requires_grad_(True)
                # It is possible that the number of iterations reaches the upper limit and none of the specified attack tags are reached
                # If so, it may need to be seen as a failed attack

        # Calculate the gradient of the average accumulation
        r = (self.a / self.iters) * total_grad
        # r = pert_image - image
        # Binary optimization
        pert_image = self.binary_optimize(output, image, image + r)
        r = pert_image - image
        # Retain some points with a certain probability
        mask = (torch.rand(image.shape) < self.cr).to(self.device)
        pert_image = torch.clamp(image + r * mask, 0, 1)

        return pert_image

    def gradient_adjust(self, new_image, loss, attack_target):
        """
        Gradient adjustment
        :param new_image:     X(t)
        :param loss:          Loss(t-1)
        :param attack_target: Tag of the attack
        :return:
        """
        # Get the output of the new sample
        new_image = new_image.clone().detach().requires_grad_(True)
        new_output = self.model(new_image)
        self.model.zero_grad()
        new_loss = self.criterion(new_output, attack_target)
        new_loss.backward(retain_graph=True)
        # Acquire losses and gradients for new samples
        if new_loss < loss:
            # Gradient descent
            return new_output, -new_image.grad.data.clone(), new_loss
        else:
            # Gradient ascent
            return new_output, new_image.grad.data.clone(), new_loss

    def binary_optimize(self, output, l_image, r_image):
        """
        Binary optimization
        :param output:  Output of adversarial samples
        :param l_image: Original sample
        :param r_image: Adversarial sample
        :return:
        """
        # Define thresholds for relative and absolute errors
        rtol = 0.01
        atol = 0.01
        # Less than a certain error to end the loop
        while not torch.isclose(l_image, r_image, rtol=rtol, atol=atol).all():
            m_image = (l_image + r_image) / 2
            m_output = self.model(m_image)
            # When sub_l_image and sub_r_image are close to equal, they are returned
            if m_output.argmax(1) != output.argmax(1):
                # If it is not the same as the correct label, the disturbance will be reduced
                r_image = m_image
            else:
                # If it is the same as the correct label, the perturbation will be amplified
                l_image = m_image

        return 2 * r_image - l_image
