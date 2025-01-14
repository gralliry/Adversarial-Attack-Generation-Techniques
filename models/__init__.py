# https://github.com/kuangliu/pytorch-cifar
# ------------------Select the model to train------------------

# from .vgg import *
# from .dpn import *
# from .lenet import *
# from .senet import *
# from .pnasnet import *
# from .densenet import *
# from .googlenet import *
# from .shufflenet import *
# from .shufflenetv2 import *
from .resnet import *
# from .resnext import *
# from .preact_resnet import *
# from .mobilenet import *
# from .mobilenetv2 import *
# from .efficientnet import *
# from .regnet import *
# from .dla_simple import *
# from .dla import *


# IndentifyModel = SimpleDLA
# IndentifyModel = VGG # ('VGG19')
IndentifyModel = ResNet18
# IndentifyModel = PreActResNet18
# IndentifyModel = GoogLeNet
# IndentifyModel = DenseNet121
# IndentifyModel = ResNeXt29_2x64d
# IndentifyModel = MobileNet
# IndentifyModel = MobileNetV2
# IndentifyModel = DPN92
# IndentifyModel = ShuffleNetG2
# IndentifyModel = SENet18
# IndentifyModel = ShuffleNetV2 # (1)
# IndentifyModel = EfficientNetB0
# IndentifyModel = RegNetX_200MF
# IndentifyModel = SimpleDLA
