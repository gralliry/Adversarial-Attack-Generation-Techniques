# -*- coding: utf-8 -*-
# @Time    : 2024/1/7 19:42
# @Author  : Jianye Liang
# @File    : parameter.py
# @Description :
import torch
import numpy


class BaseModel:
    def __init__(self, model, cuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

        self.model = model.to(self.device)

    def forward(self, image):
        return self.model(image)

    def attack(self, *args, **kwargs):
        raise NotImplementedError

    def test_attack_args(self, image, target, **kwargs):
        """
        攻击测试参数，返回参数必须对应self.attack()中的参数
        :param image: 原图像
        :param target: 正确的标签值
        :param kwargs: 其他参数
        :return: 返回参数对应self.attack()中的参数
        """
        return image, target

    def test_attack(self, model, dataloader, max_counter=999999, **kwargs) -> float:
        """
        测试模型正确率
        :param model: 识别模型
        :param dataloader: 数据加载器
        :param max_counter: 最大测试次数
        :return:
        """
        model = model.to(self.device)
        # 计数器
        counter = 0
        max_counter = min(max_counter, len(dataloader))
        print(f"Total Test Num: {max_counter}")
        batch_size = dataloader.batch_size
        # 整体正确率
        total_num = 0
        total_accuracy = 0
        self.model.eval()

        for image, target in dataloader:
            image = image.to(self.device)
            target = target.to(self.device)

            # 生成攻击图像
            pert_image = self.attack(*self.test_attack_args(image, target, **kwargs))

            output = model(pert_image)

            counter += 1
            total_num += batch_size
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy += accuracy

            print(f"No.{counter} Test, Accuracy: {accuracy / batch_size}")

            if counter >= max_counter:
                break

        print(f"{self.__class__.__name__}正确率: {total_accuracy / (max_counter * batch_size)}")

        return total_accuracy / (max_counter * batch_size)

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
