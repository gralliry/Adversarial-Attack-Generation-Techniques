#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description:
from .resnet import ResNet18

IndentifyModel = ResNet18

# ------------------Select the model to train------------------
"""
from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *
from .dla_simple import *
from .dla import *
"""

"""
model = models.SimpleDLA()
model = models.VGG('VGG19')
model = models.ResNet18()
model = models.PreActResNet18()
model = models.GoogLeNet()
model = models.DenseNet121()
model = models.ResNeXt29_2x64d()
model = models.MobileNet()
model = models.MobileNetV2()
model = models.DPN92()
model = models.ShuffleNetG2()
model = models.SENet18()
model = models.ShuffleNetV2(1)
model = models.EfficientNetB0()
model = models.RegNetX_200MF()
model = models.SimpleDLA()
"""
