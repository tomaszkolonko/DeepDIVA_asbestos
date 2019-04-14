"""
Model definition adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""
import logging
import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['VGG', 'vgg13_bn_a', 'vgg13_bn_b', 'vgg13_bn_c', 'vgg13_bn_d', 'vgg13_bn_e',
           'vgg13_bn_f', 'vgg13_bn_g', 'vgg13_bn_h', 'vgg13_bn_i', 'vgg13_bn_j']

model_urls = {'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth'}


class VGG(nn.Module):

    def __init__(self, final_filters, fc, features, output_channels=1000):
        super(VGG, self).__init__()

        self.expected_input_size = (224, 224)

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(final_filters * 7 * 7, fc),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fc, fc),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fc, output_channels),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'B': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'C': [16, 16, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'D': [16, 16, 'M', 16, 16, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M'],
    'E': [16, 16, 'M', 16, 16, 'M', 16, 16, 'M', 16, 16, 'M', 16, 16, 'M'],
    'F': [8, 8, 'M', 8, 8, 'M', 8, 8, 'M', 8, 8, 'M', 8, 8, 'M'],
    'G': [4, 4, 'M', 4, 4, 'M', 4, 4, 'M', 4, 4, 'M', 4, 4, 'M'],
    'H': [2, 2, 'M', 2, 2, 'M', 2, 2, 'M', 2, 2, 'M', 2, 2, 'M'],
    'I': [2, 2, 'M', 4, 4, 'M', 8, 8, 'M', 16, 16, 'M', 32, 32, 'M'],
    'J': [32, 32, 'M', 16, 16, 'M', 8, 8, 'M', 4, 4, 'M', 2, 2, 'M']
}

def vgg13_bn_a(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    final_filters = 256
    fully_connected_layer_size = 16
    model = VGG(final_filters, fully_connected_layer_size, make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_bn_b(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    final_filters = 128
    fully_connected_layer_size = 16
    model = VGG(final_filters, fully_connected_layer_size, make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_bn_c(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    final_filters = 64
    fully_connected_layer_size = 4096
    model = VGG(final_filters, fully_connected_layer_size, make_layers(cfg['C'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_bn_d(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    final_filters = 32
    fully_connected_layer_size = 16
    model = VGG(final_filters, fully_connected_layer_size, make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_bn_e(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    final_filters = 16
    fully_connected_layer_size = 4096
    model = VGG(final_filters, fully_connected_layer_size, make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_bn_f(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    final_filters = 8
    fully_connected_layer_size = 4096
    model = VGG(final_filters, fully_connected_layer_size, make_layers(cfg['F'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_bn_g(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    final_filters = 4
    fully_connected_layer_size = 16
    model = VGG(final_filters, fully_connected_layer_size, make_layers(cfg['G'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_bn_h(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    final_filters = 2
    fully_connected_layer_size = 4096
    model = VGG(final_filters, fully_connected_layer_size, make_layers(cfg['H'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_bn_i(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    final_filters = 32
    fully_connected_layer_size = 4096
    model = VGG(final_filters, fully_connected_layer_size, make_layers(cfg['I'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_bn_j(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    final_filters = 2
    fully_connected_layer_size = 16
    model = VGG(final_filters, fully_connected_layer_size, make_layers(cfg['J'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model