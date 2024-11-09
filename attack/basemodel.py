# -*- coding: utf-8 -*-
# @Description:
import torch
import numpy


class BaseModel:
    def __init__(self, model, cuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.model = model.to(self.device)

    def forward(self, image):
        return self.model(image)

    def gen_target(self, target):
        """
        攻击测试参数，返回参数必须对应self.attack()中的参数
        :param target: 正确的标签值
        :param kwargs: 其他参数
        :return: 返回参数对应self.attack()中的参数
        """
        raise NotImplementedError

    def attack(self, image, target):
        raise NotImplementedError

    def totensor(self, tensor, requires_grad=False) -> torch.Tensor:
        """
        自动转换/深复制为tensor
        :param tensor: 要转换/复制的张量
        :param requires_grad: 是否启动梯度计算
        :return:
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.clone().detach().to(self.device).requires_grad_(requires_grad)
        elif isinstance(tensor, numpy.ndarray):
            return torch.from_numpy(tensor).to(self.device).requires_grad_(requires_grad)
        else:
            return torch.tensor(tensor, device=self.device, requires_grad=requires_grad)
