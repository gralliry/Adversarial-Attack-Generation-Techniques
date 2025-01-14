# 对抗攻击生成技术

## 文件概述

```
│
├── attack  # 包含攻击模型类的文件夹
│    └── fgsm.py
│    └── ...
│
├── datasets          # 数据集文件夹
│    └── download.py  # 用于下载数据集的脚本
│
├── parameter         # 模型参数文件夹
│    └── ResNet
│    │    └── ...pth
│    └── ...
│    │ 
│    └── UPSET             # UPSET 使用的扰动生成模型参数文件夹
│         └── target_0     # 对应于扰动生成模型参数的目标文件夹
│         │    └── ...pth
│         └── target_1
│         └── ...
│
├── models       
│    └── resnet.py  # 识别模型类
│    └── ...
│
├── report                   # 用于攻击效果结果的文件夹
│ 
├── tensorboard              # Tensorboard 训练过程日志文件夹
│
├── contrast.py              # 攻击效果的可视化，需要图形界面，无图形化
│
├── test.py                  # 攻击后测试准确性
│
├── test_num_workers.py      # 测试最佳工作线程数量
│
├── train.py                 # 训练识别模型
│
└── train_residual_model.py  # 训练 UPSET 方法的扰动生成模型
```

## 安装

* Python-3.10 + Cuda-11.7 + Cudnn-8800

```shell
git clone https://github.com/gralliry/Adversarial-Attack-Generation-Techniques.git
cd Adversarial-Attack-Generation-Techniques
pip install -r requirements.txt
```

## 训练识别模型

```shell
# -e --epoch 指定训练 epoch 的数量，可选，默认为 100
python train.py -e 100
```

## 可视化攻击效果

```shell
# -m 指定攻击方法
# L-BFGS, FGSM, I-FGSM, JSMA, ONE-PIXEL, C&W, DEEPFOOL, MI-FGSM, UPSET
python contrast.py -m FGSM
```

## 攻击后测试准确性

```shell
# -m 指定攻击方法
# L-BFGS、FGSM、I-FGSM、JSMA、单像素、c&w、deepfool、mi-fgsm、翻转
# -c 指定测试运行次数，可选，默认值为 1000
python test.py -m FGSM -c 1000
```

## 例外：训练 UPSET 方法的扰动生成模型

UPSET方法需要单独训练一个扰动生成模型

```shell
# -t --target 指定扰动生成模型的目标标签
# 范围是 0 ~ 9，对应飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
# -e --epoch 指定训练 epoch 的数量，可选，默认为 100
# -lr --learning_rate 指定学习率，可选，默认为 1e-3

# 训练扰动模型需要一个已经训练的识别模型，请在脚本中修改或加载参数文件
python train_residual_model.py -t 0 -e 100
```

## 测试最佳工作线程数

```shell
# 运行并根据执行时间选择最佳工作线程数
python test_num_workers.py
```

## 贡献者

- Liang Jianye - SCAU
