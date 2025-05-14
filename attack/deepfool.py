# -*- coding: utf-8 -*-
# @Description:
import torch

from .base import BaseModel


class DeepFool(BaseModel):
    def __init__(self, model, overshoot=0.001, iters=10, cuda=True):
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

    def attack(self, image, target, is_targeted=False):
        """
        DeepFool
        Only data with batch_size = 1 will be accepted
        :param image:  Tensors that need to be processed
        :param target: the target is useless, leave it alone
        :param is_targeted: the is_targeted is useless, leave it alone
        :return:       Adversarial samples generated
        """
        assert image.size(0) == 1, ValueError("Only data with batch_size = 1 will be accepted")
        assert is_targeted is False, ValueError("DeepFool must be not targeted")
        # pert_image = self.totensor(image)
        x = image.clone().detach().requires_grad_(True)
        # The gradient gap from the current label to the correct label
        w = torch.zeros_like(image)
        r_tot = torch.zeros_like(image)
        # Get the right predicted output
        f_image = self.model(image)
        # Get index sorting for the correctness of all labels
        i_classes = f_image[0].argsort().cpu().numpy()[::-1]

        with torch.enable_grad():
            for _ in range(self.iters):
                # Gets the label of the image after adding the perturbation
                fs = self.model(x)
                k_i = fs.argmax(dim=1)
                # If the label is still the correct label after the prediction
                if k_i != target:
                    # Successfully generate adversarial samples and exit directly
                    break
                # Set the minimum loss to 0
                pert = 0
                # Backpropagation for the correct labels
                fs[0, i_classes[0]].backward(retain_graph=True)
                # The gradient of the current correct label
                grad_orig = x.grad.data.clone().detach()
                for k in range(1, i_classes.shape[0]):
                    # Backpropagation for each corresponding label
                    fs[0, i_classes[k]].backward(retain_graph=True)
                    # The gradient of the current label
                    cur_grad = x.grad.data.clone().detach()
                    # Calculate the gradient gap from the current label to the correct label
                    w_k = cur_grad - grad_orig
                    # Calculates the difference between the current label output and the correct label output
                    f_k = fs[0, i_classes[k]] - fs[0, i_classes[0]]
                    # Calculate the normalized perturbation magnitude
                    pert_k = f_k.abs() / torch.linalg.norm(w_k.flatten())
                    # Select the larger loss and replace
                    if pert_k > pert:
                        pert = pert_k
                        w = w_k

                # Prevent a pert of 0
                # from normalizing the magnitude of the perturbation and keep it in the direction of gradient w
                r_tot = r_tot + (pert + 1e-7) * w / torch.linalg.norm(w)

                # Add perturbations to the original image to generate adversarial samples
                x = image + (1 + self.overshoot) * r_tot
                x = torch.clamp(x, 0, 1).detach().requires_grad_(True)

        return x.detach()
