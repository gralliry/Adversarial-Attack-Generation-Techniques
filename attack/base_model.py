# -*- coding: utf-8 -*-
# @Description:
import torch


class BaseModel:
    def __init__(self, model, cuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def forward(self, image):
        return self.model(image)

    def attack(self, image, target):
        raise NotImplementedError
