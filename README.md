# Adversarial Attack Generation Techniques

[中文版本](./assets/README.cn.md)

## File overview

```
│
├── attack  # The folder that contains the attack model class
│    └── fgsm.py
│    └── ...
│
├── datasets          # Dataset folders
│    └── download.py  # A script for downloading the dataset
│
├── parameter         # Model Parameters folder
│    └── ResNet
│    │    └── ...pth
│    └── ...
│    │ 
│    └── UPSET             # The perturbation used by UPSET generates a folder of model parameters
│         └── target_0     # Corresponds to the destination folder for perturbation generation model parameters
│         │    └── ...pth
│         └── target_1
│         └── ...
│
├── models       
│    └── resnet.py  # Identify model classes
│    └── ...
│
├── report                   # The folder used for the results of the attack effect
│ 
├── tensorboard              # Tensorboard Training process log folder
│
├── contrast.py              # The visualization of the attack effect requires a graphical interface
│
├── test.py                  # Test post-attack accuracy
│
├── test_num_workers.py      # Test the optimal number of worker threads
│
├── train.py                 # Train a recognition model
│
└── train_residual_model.py  # Train a perturbation generation model for the UPSET method
```

## Install

* Python-3.10 + Cuda-11.7 + Cudnn-8800

```shell
git clone https://github.com/gralliry/Adversarial-Attack-Generation-Techniques.git
cd Adversarial-Attack-Generation-Techniques
pip install -r requirements.txt
```

## Train the recognition model

```shell
# -e --epoch Specify the number of training epochs, optional, defaults to 100
python train.py -e 100
```

## Visualize the impact of the attack

```shell
# -m Specify the attack method
# L-BFGS, FGSM, I-FGSM, JSMA, ONE-PIXEL, C&W, DEEPFOOL, MI-FGSM, UPSET
python contrast.py -m FGSM
```

## Test post-attack accuracy

```shell
# -m Specify the attack method
# L-BFGS、FGSM、I-FGSM、JSMA、单像素、c&w、deepfool、mi-fgsm、翻转
# -c Specify the number of test runs, optional, defaults to 1000
python test.py -m FGSM -c 1000
```

## Exception: Perturbation generation model for training the UPSET method

The UPSET method requires a perturbation generation model to be trained separately

```shell
# -t --target Specifies the target label for the perturbation generation model
# The range is 0 ~ 9, corresponding to airplanes, cars, birds, cats, deer, dogs, frogs, horses, boats, trucks
# -e --epoch Specify the number of training epochs, optional, defaults to 100
# -lr --learning_rate Specify the learning rate, optional, defaults to 1e-3

# Training a perturbation model requires a trained recognition model, which can be modified or loaded in the script
python train_residual_model.py -t 0 -e 100
```

## Test the optimal number of worker threads

```shell
# Run and select the optimal number of worker threads based on execution time
python test_num_workers.py
```

## Contributors

- Chen Yiyi
- Americ Garcia
