# -*- coding: utf-8 -*-
# @Description:
import torch
import numpy as np

from .base import BaseModel


class JSMA(BaseModel):
    def __init__(self, model, theta=1.0, gamma=0.1, cuda=True):
        """
        JSMA

        https://arxiv.org/abs/1511.07528

        https://github.com/probabilistic-jsmas/probabilistic-jsmas

        https://github.com/guidao20/MJSMA_JSMA/blob/master/MJSMA_JSMA.py

        https://github.com/FenHua/Adversarial-Examples/blob/master/%E9%BB%91%E7%9B%92/JSMA/JSMA.ipynb
        :param model:
        :param theta: Perturbation step size
        :param gamma: Define the limits of change/boundaries
        :param cuda:  Whether to start CUDA
        """
        super().__init__(model=model, cuda=cuda)

        self.theta = theta
        self.gamma = gamma

    def attack(self, image, target, is_targeted=True):
        """
        JSMA
        :param image:
        :param target: Tag of the attack
        :param is_targeted: is_targeted
        :return:
        """
        assert image.size(0) == 1, ValueError("Only data with batch_size = 1 will be accepted")
        assert is_targeted is True, ValueError("JSMA must be targeted")

        var_image = image.clone().detach().requires_grad_(True)
        if self.theta > 0:
            increasing = True
        else:
            increasing = False

        num_features = int(np.prod(var_image.shape[1:]))
        shape = var_image.shape

        # Perturb two pixels in one iteration, thus max_iters is divided by 2
        max_iters = int(np.ceil(num_features * self.gamma / 2.0))

        # Masked search domain, if the pixel has already reached the top or bottom, we don't bother to modify it
        if increasing:
            domain = torch.lt(var_image, 0.99)
        else:
            domain = torch.gt(var_image, 0.01)

        domain = domain.view(num_features)
        output = self.model(var_image)
        current_pred = torch.argmax(output, 1)
        iters = 0
        while (
                (iters < max_iters)
                and (current_pred != target)
                and (domain.sum() != 0)
        ):
            # Calculate Jacobian matrix of forward derivative
            jacobian = self.compute_jacobian(var_image)
            # Get the saliency map and calculate the two pixels that have the greatest influence
            p1, p2 = self.saliency_map(jacobian, target, increasing, domain, num_features)
            # Apply modifications
            var_sample_flatten = var_image.view(-1, num_features).clone().detach()
            # var_sample_flatten = var_image.view(-1, num_features)
            var_sample_flatten[0, p1] += self.theta
            var_sample_flatten[0, p2] += self.theta

            new_image = torch.clamp(var_sample_flatten, min=0.0, max=1.0)
            new_image = new_image.view(shape)
            domain[p1] = 0
            domain[p2] = 0
            var_image = new_image.clone().detach().to(self.device)

            output = self.model(var_image)
            current_pred = torch.argmax(output.data, 1)
            iters += 1

        return var_image.detach()

    def compute_jacobian(self, image):
        var_image = image.clone().detach().requires_grad_(True)
        output = self.model(var_image)

        num_features = int(np.prod(var_image.shape[1:]))
        jacobian = torch.zeros([output.shape[1], num_features])
        for i in range(output.shape[1]):
            if var_image.grad is not None:
                var_image.grad.zero_()
            output[0][i].backward(retain_graph=True)
            # Copy the derivative to the target place
            jacobian[i] = (
                var_image.grad.squeeze().view(-1, num_features).clone()
            )  # nopep8

        return jacobian.to(self.device)

    # Calculate saliency plots
    @torch.no_grad()
    def saliency_map(self, jacobian, target_label, increasing, search_space, nb_features):
        # The search domain
        domain = torch.eq(search_space, 1).float()
        # The sum of all features' derivative with respect to each class
        all_sum = torch.sum(jacobian, dim=0, keepdim=True)
        # The forward derivative of the target class
        target_grad = jacobian[target_label]
        # The sum of forward derivative of other classes
        others_grad = all_sum - target_grad

        # This list blanks out those that are not in the search domain
        if increasing:
            increase_coef = 2 * (torch.eq(domain, 0)).float().to(self.device)
        else:
            increase_coef = -1 * 2 * (torch.eq(domain, 0)).float().to(self.device)
        increase_coef = increase_coef.view(-1, nb_features)

        # Calculate sum of target forward derivative of any 2 features.
        target_tmp = target_grad.clone()
        target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
        # PyTorch will automatically extend the dimensions
        alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(
            -1, nb_features, 1
        )
        # Calculate sum of other forward derivative of any 2 features.
        others_tmp = others_grad.clone()
        others_tmp += increase_coef * torch.max(torch.abs(others_grad))
        beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

        # Zero out the situation where a feature sums with itself
        tmp = np.ones((nb_features, nb_features), int)
        np.fill_diagonal(tmp, 0)
        zero_diagonal = torch.from_numpy(tmp).byte().to(self.device)

        # According to the definition of saliency map in the paper (formulas 8 and 9),
        # those elements in the saliency map that doesn't satisfy the requirement will be blanked out.
        if increasing:
            mask1 = torch.gt(alpha, 0.0)
            mask2 = torch.lt(beta, 0.0)
        else:
            mask1 = torch.lt(alpha, 0.0)
            mask2 = torch.gt(beta, 0.0)

        # Apply the mask to the saliency map
        mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
        # Do the multiplication according to formula 10 in the paper
        saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
        # Get the most significant two pixels
        max_idx = torch.argmax(saliency_map.view(-1, nb_features * nb_features), dim=1)
        # p = max_idx // nb_features
        p = torch.div(max_idx, nb_features, rounding_mode="floor")
        # q = max_idx % nb_features
        q = max_idx - p * nb_features
        return p, q
