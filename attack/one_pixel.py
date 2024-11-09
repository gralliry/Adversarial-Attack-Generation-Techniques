# -*- coding: utf-8 -*-
# @Description:
import torch
import numpy as np
from torch.functional import F
from .base_model import BaseModel


class ONE_PIXEL(BaseModel):
    def __init__(self, model, pixels_size=40, pop_size=30, iters=15, cr=0.75, factor=0.5, pixels_changed=1,
                 cuda=True):
        """
        ONE_PIXEL

        https://arxiv.org/abs/1710.08864

        https://github.com/DebangLi/one-pixel-attack-pytorch

        https://github.com/Hyperparticle/one-pixel-attack-keras
        :param model: 模型
        :param iters: 迭代次数
        :param pixels_size: 预选的像素数量/种群个数
        :param pop_size: 一个像素中种群个体数量
        :param cr: 交叉重组概率
        :param factor: 缩放因子 N = a + F*(b-c)，产生变异个体
        :param pixels_changed: 改变的像素数量
        :param cuda: 是否启动cuda
        """
        super().__init__(model=model, cuda=cuda)

        self.pixels_size = pixels_size
        self.pop_size = pop_size
        self.iters = iters
        self.cr = cr
        self.factor = factor
        self.pixels_changed = pixels_changed

    def perturb(self, image, pos, rgb):
        pert_image = image.clone().detach().requires_grad_(True)
        # 根据个体生成 新的 对抗样本
        for i in range(3):
            pert_image[0, i, pos[0], pos[1]] = rgb[i]

        return pert_image

    def evaluate(self, pos_candidates, rgb_candidates, img, label):
        # 所有像素点种群的个体的适应度
        fitness = []
        with torch.no_grad():
            # 遍历每个像素的种群
            for index in range(self.pixels_size):
                # 获取像素的坐标
                pos = pos_candidates[index]
                # 单个像素点的种群的个体适应度
                pixel_fitness = []
                # 遍历每个种群的个体
                for rgb in rgb_candidates[index]:
                    # 生成 新的 对抗样本
                    pert_image = self.perturb(img, pos, rgb)
                    # 获取新图像的输出
                    output = self.model(pert_image).squeeze()
                    # 概率归一化 # F.softmax归一化的概率差距太大，容易造成遗漏
                    softmax = F.sigmoid(output)
                    # 选择对应标签的适应度
                    pixel_fitness.append(softmax[label[0]].item())
                fitness.append(pixel_fitness)
        return np.array(fitness)

    def evolve(self, rgb_candidates):
        """
        演化，在演化的过程，由于不同像素不同的RGB的改变都会影响loss
        在演化时，必须针对属于该像素的群体演化，不能使用其他像素的群体进行演化
        需要为每个被选中的像素点创建群体，并在属于该橡素点的群体中进行演化
        :param rgb_candidates:
        :return:
        """
        gen = rgb_candidates.copy()
        # 遍历每个像素的种群
        for index, pixel_candidates in enumerate(rgb_candidates):
            pixel_candidates = pixel_candidates.copy()
            # 遍历每个种群的个体/候选解 # 对每个候选解进行演化
            for i in range(self.pop_size):
                # 从当前候选解中随机选择3个不同的解
                x1, x2, x3 = pixel_candidates[np.random.choice(self.pop_size, 3, replace=False)]
                # 差分操作，生成新的解
                next_rgb = x1 + self.factor * (x2 - x3)
                # 选择性遗传，概率为 cr
                choose = np.random.choice((True, False), size=3, p=(self.cr, 1 - self.cr))
                next_rgb[choose] = pixel_candidates[i][choose]
                # 处理越界值
                x_oob = np.logical_or((next_rgb < 0), (1 < next_rgb))
                # 将越界的值替换为新的随机值
                next_rgb[x_oob] = np.random.random(3)[x_oob]
                # 将生成的新解放入下一代中
                pixel_candidates[i] = next_rgb
            gen[index] = pixel_candidates
        return gen

    def attack(self, image, target, is_targeted=False):
        assert image.size(0) == 1, ValueError("只接受 batch_size = 1 的数据")
        # 生成欺骗标签
        # 这里只是单纯生成错误的标签，并没有指定标签，所以攻击后识别成功率还是会偏高
        # (target + 1) % 10
        image = image.clone().detach().requires_grad_(True)
        # 使用均匀分布 X~U(0,31) Y~U(0,31) 来生成 X, Y
        coordinates = np.mgrid[0:image.size(2), 0:image.size(3)].reshape(2, -1).T
        pos_candidates = coordinates[np.random.choice(coordinates.shape[0], self.pixels_size, replace=False)]
        # pos_candidates = np.random.randint(0, image.size(2), size=(self.pixels_size, 2))
        # 使用正态分布 R~N(0.5, 0.5) G~N(0.5, 0.5) B~N(0.5, 0.5) 来生成 R, G, B ?
        rgb_candidates = np.random.normal(0.5, 0.5, size=(self.pixels_size, self.pop_size, 3))
        # 评估每个候选解的适应度
        fitness = self.evaluate(pos_candidates, rgb_candidates, image, target)
        # 迭代过程
        for _ in range(self.iters):
            # 生成新的候选解
            new_rgb_candidates = self.evolve(rgb_candidates)
            # 计算新的适应度
            new_fitness = self.evaluate(pos_candidates, new_rgb_candidates, image, target)
            # 根据是否有目标攻击，选择更高或更低的适应度来更新种群
            successors = (new_fitness > fitness) if is_targeted else (new_fitness < fitness)
            rgb_candidates[successors] = new_rgb_candidates[successors]
            fitness[successors] = new_fitness[successors]

            # 如果适应度满足过大(有目标)/过小(无目标)，可看作攻击成功，直接退出迭代
            if (is_targeted and np.max(fitness) > 0.5) or (not is_targeted and np.min(fitness) < 0.05):
                break

        # 对根据适应度对种群及个体进行排序
        if is_targeted:
            # 从大到小
            pixels_fitness_arg = np.array([np.argmax(subfitness) for subfitness in fitness])
            pixels_fitness = np.array([np.max(subfitness) for subfitness in fitness])
            indexarr = np.argsort(pixels_fitness)[::-1]
        else:
            # 从小到大
            pixels_fitness_arg = np.array([np.argmin(subfitness) for subfitness in fitness])
            pixels_fitness = np.array([np.min(subfitness) for subfitness in fitness])
            indexarr = np.argsort(pixels_fitness)
        # 根据最好的适应度进行选择并叠加在原样本中生成对抗样本
        perturb_img = image
        for index in range(min(self.pixels_size, self.pixels_changed)):
            perturb_img = self.perturb(perturb_img,
                                       pos_candidates[indexarr[index]],
                                       rgb_candidates[indexarr[index], pixels_fitness_arg[index]],
                                       )
        return perturb_img
