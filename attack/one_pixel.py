# -*- coding: utf-8 -*-
# @Description:
import torch
import numpy as np
from torch.functional import F
from .base import BaseModel


class ONE_PIXEL(BaseModel):
    def __init__(self, model, pixels_size=40, iters=15, cr=0.75, factor=0.5, pixels_changed=1, cuda=True):
        """
        ONE_PIXEL

        https://arxiv.org/abs/1710.08864

        https://github.com/DebangLi/one-pixel-attack-pytorch

        https://github.com/Hyperparticle/one-pixel-attack-keras
        :param model: 模型
        :param iters: 迭代次数
        :param pixels_size: 预选的像素数量/种群个数
        :param cr: 交叉重组概率
        :param factor: 缩放因子 N = a + F*(b-c)，产生变异个体
        :param pixels_changed: 改变的像素数量
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)

        self.pixels_size = pixels_size
        self.iters = iters
        self.cr = cr
        self.factor = factor
        self.pixels_changed = pixels_changed

    @torch.no_grad()
    def perturb(self, image, vi):
        # image [1, c, h, w]
        pert_image = image.clone().detach()
        # print(pert_image.shape, vi.shape)
        # 根据个体生成 新的 对抗样本
        pert_image[0, :, int(vi[0] * image.shape[2]), int(vi[1] * image.shape[3])] = torch.from_numpy(vi[2:5]).to(
            self.device)
        return pert_image

    def evaluate(self, vs: np.ndarray, img, label) -> np.ndarray:
        # 所有像素点种群的个体的适应度
        fitnesses = []
        # 遍历每个像素的种群
        for index in range(self.pixels_size):
            # 生成 新的 对抗样本
            pert_image = self.perturb(img, vs[index])
            # 获取适应度
            fitness = self.fitness(pert_image, label)
            fitnesses.append(fitness)
        return np.array(fitnesses)

    @torch.no_grad()
    def fitness(self, image, label):
        # 获取新图像的输出
        output = self.model(image).squeeze()
        # 概率归一化 # F.softmax归一化的概率差距太大，容易造成遗漏
        softmax = F.sigmoid(output)
        return softmax[label[0]].item()

    def evolve(self, vs):
        """
        演化，在演化的过程，由于不同像素不同的RGB的改变都会影响loss
        在演化时，必须针对属于该像素的群体演化，不能使用其他像素的群体进行演化
        需要为每个被选中的像素点创建群体，并在属于该橡素点的群体中进行演化
        :param vs:
        :return:
        """
        vs = vs.copy()
        # 遍历每个像素的种群
        for index in range(vs.shape[0]):
            # 从当前候选解中随机选择3个不同的解
            x1, x2, x3 = vs[np.random.choice(self.pixels_size, 3, replace=False)]
            # 差分操作，生成新的解
            next_v = x1 + self.factor * (x2 - x3)
            # 处理越界值
            next_v = np.clip(next_v, 0, 1)
            # 选择性遗传，概率为 cr
            choose = np.random.choice((True, False), size=5, p=(self.cr, 1 - self.cr))
            vs[index][choose] = next_v[choose]
        return vs

    def attack(self, image, target, is_targeted=False):
        assert image.size(0) == 1, ValueError("只接受 batch_size = 1 的数据")
        image = image.clone().detach().requires_grad_(True)
        # 使用均匀分布 X~U(0,31) Y~U(0,31) 来生成 X, Y
        x_y = np.random.uniform(0.0, 1.0, size=(self.pixels_size, 2))
        # 生成RGB值（方法1：直接生成0-255范围）
        rgb = np.random.normal(loc=0.5, scale=0.5, size=(self.pixels_size, 3))
        rgb = np.clip(rgb, 0, 1)

        vs = np.hstack((x_y, rgb))
        # 评估每个候选解的适应度
        fitness = self.evaluate(vs, image, target)
        # 迭代过程
        for _ in range(self.iters):
            # 生成新的候选解
            new_vs = self.evolve(vs)
            # 计算 新 的 适应度
            new_fitness = self.evaluate(vs, image, target)
            # 根据是否有目标攻击，选择更高或更低 的 适应度来更新种群
            successors = (new_fitness > fitness) if is_targeted else (new_fitness < fitness)
            vs[successors] = new_vs[successors]
            fitness[successors] = new_fitness[successors]

            # 如果适应度满足过大(有目标)/过小(无目标)，可看作攻击成功，直接退出迭代
            if (is_targeted and np.max(fitness) > 0.5) or (not is_targeted and np.min(fitness) < 0.05):
                break

        # 对根据适应度对种群及个体进行排序
        # 从大到小  # 从小到大
        indexarr = np.argsort(fitness)[::-1] if is_targeted else np.argsort(fitness)
        # 根据d最好 的 适应度 进行选择并叠加在原样本中生成对抗样本
        perturb_img = image
        for index in range(min(self.pixels_size, self.pixels_changed)):
            perturb_img = self.perturb(perturb_img, vs[indexarr[index]])
        return perturb_img
