# Adversarial Attack Generation Techniques

## File Overview

```
│
├── attack  # Folder containing attack model classes
│    └── fgsm.py
│    └── ...
│
├── datasets          # Folder for datasets
│    └── download.py  # Script to download datasets
│
├── parameter         # Folder for model parameters
│    └── ResNet
│    │    └── ...pth
│    └── ...
│    │ 
│    └── UPSET          # Folder for perturbation generation model parameters used by UPSET
│         └── target_0  # Target folders corresponding to perturbation generation model parameters
│         │    └── ...pth
│         └── target_1
│         └── ...
│
├── models       
│    └── resnet.py  # Recognition model class
│    └── ...
│
├── report  # Folder for attack effect visualization
│
├── tensorboard      # Folder for tensorboard training process logs
│
├── contrast.py          # Visualization of attack effects, requires a graphical interface
│
├── test.py              # Test accuracy after attack
│
├── test_num_workers.py  # Test the optimal number of workers
│
├── train.py             # Train recognition model
│
└── train_residual_model.py  # Train the perturbation generation model for UPSET method
```
## Installation

```shell
git clone https://github.com/gralliry/Pytorch-Generative-Adversarial-Network.git
cd Pytorch-Generative-Adversarial-Network
pip install -r requirements.txt
```

## Train Recognition Model

```shell
# -e --epoch specifies the number of training epochs, optional, default is 100
python train.py -e 100
```

## Visualize Attack Effects

```shell
# -m specifies the attack method
# L-BFGS, FGSM, I-FGSM, JSMA, ONE-PIXEL, C&W, DEEPFOOL, MI-FGSM, UPSET
python contrast.py -m FGSM
```

## Test Accuracy After Attack

```shell
# -m specifies the attack method
# L-BFGS, FGSM, I-FGSM, JSMA, ONE-PIXEL, C&W, DEEPFOOL, MI-FGSM, UPSET
# -c specifies the number of test runs, optional, default is 500
python test.py -m FGSM -c 100
```

## Train Perturbation Generation Model for UPSET Method

```shell
# -t --target specifies the target label for the perturbation generation model
# Range is 0 ~ 9, corresponding to plane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# -e --epoch specifies the number of training epochs, optional, default is 100
# -lr --learning_rate specifies the learning rate, optional, default is 1e-3

# Training the perturbation model requires a recognition model, please modify or load it in the script
python train_residual_model.py -t 0 -e 100
```

## Test Optimal Number of Workers

```shell
# Run and select the best number of workers based on execution time
python test_num_workers.py
```

## Contributors

- Liang Jianye - SCAU