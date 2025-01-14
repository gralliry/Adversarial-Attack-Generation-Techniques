# -*- coding: utf-8 -*-
# @Description:
import torch

import numpy as np

from .base_model import BaseModel


class DeepFool(BaseModel):
    def __init__(self, model, overshoot=0.02, iters=50, cuda=True):
        """
        DeepFool

        https://arxiv.org/abs/1511.04599
        https://github.com/LTS4/DeepFool
        https://github.com/aminul-huq/DeepFool
        https://medium.com/@aminul.huq11/pytorch-implementation-of-deepfool-53e889486ed4
        :param model:
        :param overshoot: Maximum limit
        :param iters:     The number of iterations
        :param cuda:      Whether to start CUDA
        """
        super().__init__(model=model, cuda=cuda)

        self.overshoot = overshoot
        self.iters = iters

    def attack(self, image, target):
        """
        # DeepFool
        Only data with batch_size = 1 will be accepted
        :param image:  Tensors that need to be processed
        :param target: the target is useless, leave it alone (hhhh)
        :return:       Adversarial samples generated
        """
        assert image.size(0) == 1, ValueError("Only data with batch_size = 1 will be accepted")
        image = image.clone().detach()
        # pert_image = self.totensor(image)
        x = image.clone().detach().requires_grad_(True)
        # Get the right predicted output
        f_image = self.model(image)
        fs = self.model(x)
        # Get the number of categories/tags
        num_classes = f_image.size(1)
        # Get index sorting for the correctness of all labels
        i_classes = f_image.data[0].argsort().cpu().numpy()[::-1]
        i_classes = i_classes[0:num_classes]
        # The prediction label of the current model output
        label = i_classes[0]
        k_i = label
        # The gradient gap from the current label to the correct label
        w = np.zeros(image.shape)
        r_tot = np.zeros(image.shape)

        self.model.eval()
        with torch.set_grad_enabled(True):
            for _ in range(self.iters):
                # If the label is still the correct label after the prediction
                if k_i != label:
                    # Successfully generate adversarial samples and exit directly
                    break
                # Set the minimum loss to 0
                pert = 0
                # Backpropagation for the correct labels
                fs[0, i_classes[0]].backward(retain_graph=True)
                # The gradient of the current correct label
                grad_orig = x.grad.data.cpu().numpy().copy()

                for k in range(1, num_classes):
                    # Clear Gradient # Prevent the gradient of other labels from affecting the calculation of the current label gradient
                    if x.grad is not None:
                        x.grad.zero_()
                    # Backpropagation for each corresponding label
                    fs[0, i_classes[k]].backward(retain_graph=True)
                    # The gradient of the current label
                    cur_grad = x.grad.data.cpu().numpy().copy()
                    # Calculate the gradient gap from the current label to the correct label
                    w_k = cur_grad - grad_orig
                    # Calculates the difference between the current label output and the correct label output
                    f_k = (fs[0, i_classes[k]] - fs[0, i_classes[0]]).data.cpu().numpy()
                    # Calculate the normalized perturbation magnitude
                    pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
                    # Select the larger loss and replace
                    if pert_k > pert:
                        pert = pert_k
                        w = w_k

                # Prevent a pert of 0 # from normalizing the magnitude of the perturbation and keep it in the direction of gradient w
                r_i = (pert + 1e-4) * w / np.linalg.norm(w)
                r_tot = np.float32(r_tot + r_i)

                # Add perturbations to the original image to generate adversarial samples
                x = image + (1 + self.overshoot) * torch.from_numpy(r_tot).to(self.device)
                x.requires_grad = True

                # Gets the label of the image after adding the perturbation
                fs = self.model(x)
                k_i = np.argmax(fs.data.cpu().numpy().flatten())

        return x
