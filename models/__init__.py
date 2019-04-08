# Keep the list of models implemented up-2-date
from .CNN_basic import CNN_basic
from .FC_medium import FC_medium
from .FC_simple import FC_simple
from .TNet import TNet
from .AlexNet import alexnet
from .ResNet import resnet18, resnet34, resnet50, resnet101, resnet152

# DIFFERENT FILTERS IN ARCHITECTURE
# **********************************
from models.ResNet_Filters.ResNet18_Filter_One import resnet_one
from models.ResNet_Filters.ResNet18_Filter_Two import resnet_two
from models.ResNet_Filters.ResNet18_Filter_Four import resnet_four
from models.ResNet_Filters.ResNet18_Filter_Eight import resnet_eigth
from models.ResNet_Filters.ResNet18_Filter_Sixteen import resnet_sixteen
from models.ResNet_Filters.ResNet18_Filter_ThirtyTwo import resnet_thirtytwo

# DIFFERENT INPUT SIZES IN ARCHITECTURE
# *************************************
from models.ResNet_Image_Input.ResNet18_448 import resnet18_448
from models.ResNet_Image_Input.ResNet18_896 import resnet18_896
from models.ResNet_Image_Input.ResNet18_1024 import resnet18_1024

# DIFFERENT FC FOR VGG13
# **********************
from models.VGG13_Variants.VGG13_FC import *

from .BabyResNet import babyresnet18, babyresnet34, babyresnet50, babyresnet101, babyresnet152
from ._VGG import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from ._Inception_v3 import inception_v3
from ._DenseNet import densenet121, densenet161, densenet169, densenet201
from .CAE_basic import CAE_basic
from .CAE_medium import CAE_medium
from .FusionNet import FusionNet
from .UNet import Unet
from .CNN_basic_tanh import CNN_basic_tanh


"""
Formula to compute the output size of a conv. layer

new_size =  (width - filter + 2padding) / stride + 1
"""
