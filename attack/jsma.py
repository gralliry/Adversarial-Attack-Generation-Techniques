# -*- coding: utf-8 -*-
# @Description:
import torch
import numpy as np

from .base_model import BaseModel


class JSMA(BaseModel):
    def __init__(self, model, alpha=3.0, gamma=3.0, iters=20, cuda=True):
        """
        JSMA

        https://arxiv.org/abs/1511.07528

        https://github.com/probabilistic-jsmas/probabilistic-jsmas

        https://github.com/guidao20/MJSMA_JSMA/blob/master/MJSMA_JSMA.py

        https://github.com/FenHua/Adversarial-Examples/blob/master/%E9%BB%91%E7%9B%92/JSMA/JSMA.ipynb
        :param model:
        :param alpha: Perturbation step size
        :param gamma: Define the limits of change/boundaries
        :param iters: Maximum number of loops/looks/number of changed pixels
        :param cuda:  Whether to start CUDA
        """
        super().__init__(model=model, cuda=cuda)

        self.alpha = alpha
        self.gamma = gamma
        self.iters = iters

    def attack(self, image, target):
        """
        JSMA
        :param image:
        :param attack_target: Tag of the attack
        :return:
        """
        assert image.size(0) == 1, ValueError("Only data with batch_size = 1 will be accepted")

        pert_image = image.clone().detach().requires_grad_(True)
        # Generate spoofed labels
        # The number of fool_target elements here should be the same as batch_size
        # This is just a simple generation of wrong tags, and no tags are specified
        attack_target = (target + 1) % 10
        # Define the search field, set the modified position to zero, and no longer calculate it next time
        mask = np.ones(pert_image.shape)
        # Evaluation mode
        self.model.eval()
        with torch.set_grad_enabled(True):
            for _ in range(self.iters):
                output = self.model(pert_image)
                # This is only appropriate for a judgment where batch_size is 1
                if output.argmax(1) == attack_target:
                    # If the attack is successful, the iteration is stopped
                    break
                # Gradient clearing
                if pert_image.grad is not None:
                    pert_image.grad.zero_()
                # Backpropagation is performed on each image
                output[0, attack_target[0]].backward(retain_graph=True)
                # Generate perturbation points and perturbation sizes
                index, pix_sign = self.saliency_map(pert_image, mask)
                # Add Perturbation to Adversarial Samples
                pert_image.data[index] += pix_sign * self.alpha * self.gamma
                # Points that have reached the limit no longer participate in the update
                if not -self.gamma <= pert_image.data[index] <= self.gamma:
                    # Limit perturbations
                    pert_image.data[index] = torch.clamp(pert_image.data[index], -self.gamma, self.gamma)
                    # The pixel corresponding to the search field is zeroed, indicating that the point is no longer involved in the calculation update
                    mask[index] = 0

        return pert_image

    # Calculate saliency plots
    @staticmethod
    def saliency_map(image, mask):
        """
        This method is a simplified version of the beta parameter and focuses on the points where the attack target contributes the most
        :param image:
        :param mask: Marker bits, which record the coordinates of the points that have been visited
        :return:
        """
        derivative = image.grad.data.cpu().numpy().copy()
        # Prediction Contribution to Attack Target # is set to 0 for searched points
        alphas = derivative * mask
        # Predict the contribution to non-attack targets
        betas = -np.ones_like(alphas)
        # Calculate the gap between positive and negative perturbations
        sal_map = np.abs(alphas) * np.abs(betas) * np.sign(alphas * betas)
        # Best pixel and perturbation direction
        index = np.argmax(sal_map)
        # Convert to (p1, p2) format
        index = np.unravel_index(index, mask.shape)
        pixel_sign = np.sign(alphas)[index]
        return index, pixel_sign
