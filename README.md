# 对抗样本生成技术

## 文件介绍

```
│
├── attack  # 攻击模型类存放文件夹
│    └── fgsm.py
│    └── ...
│
├── datasets          # 数据集存放文件夹
│    └── downlaod.py  # 数据集下载文件，运行会下载对应的数据集
│
├── parameter         # 模型参数存放文件
│    └── ResNet
│    │    └── ...pth
│    └── ...
│    │ 
│    └── UPSET          # UPSET使用的扰动生成模型参数文件夹
│         └── target_0  # 扰动生成模型参对应的target文件夹
│         │    └── ...pth
│         └── target_1
│         └── ...
│
├── models       
│    └── resnet.py  # 识别模型类
│    └── ...
│
├── report  # 攻击效果展示图
│
├── tensorboard      # tensorboard训练过程保存文件夹
│
├── contrast.py          # 攻击效果展示，需要图形化界面
│
├── test.py              # 测试攻击后准确率
│
├── test_num_workers.py  # 测试最佳num_workers数
│
├── train.py             # 训练识别模型
│
└── train_residual_model.py  # 训练UPSET方法的扰动生成模型
```

## 训练识别模型

```shell
# -e --epoch 为 训练次数, 可选，默认 为 100
python train.py -e 100
```

## 攻击效果展示

```shell
# -m 为 攻击方法
# L-BFGS, FGSM, I-FGSM, JSMA, ONE-PIXEL, C&W, DEEPFOOL, MI-FGSM, UPSET
python contrast.py -m FGSM
```

## 测试攻击后准确率

```shell
# -m 为 攻击方法
# L-BFGS, FGSM, I-FGSM, JSMA, ONE-PIXEL, C&W, DEEPFOOL, MI-FGSM, UPSET
# -c 为 测试次数，可选，默认为500
python test.py -m FGSM -c 100
```

## 训练UPSET方法的扰动生成模型

```shell
# -t  --target        为 扰动生成模型针对的标签
# 范围 0 ~ 9, 对应 plane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# -e  --epoch         为 训练次数, 可选，默认 为 100
# -lr --learning_rate 为 学习率，  可选，默认 为 1e-3

# 训练扰动模型需要识别模型，请在文件中自行修改或加载
python train_residual_model.py -t 0 -e 100
```

## 测试最佳num_workers数

```shell
# 运行后根据不同num_workers数对应运行的时间选择最佳的num_workers数
python test_num_workers.py
```
