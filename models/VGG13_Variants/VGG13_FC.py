"""
Model definition adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""
import logging
import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['VGG', 'vgg13_fc_1024', 'vgg13_fc_1024_bn',
           'vgg13_fc_512', 'vgg13_fc_512_bn',
           'vgg13_fc_256', 'vgg13_fc_256_bn',
           'vgg13_fc_128', 'vgg13_fc_128_bn',
           'vgg13_fc_64', 'vgg13_fc_64_bn',
           'vgg13_fc_32', 'vgg13_fc_32_bn',
           'vgg13_fc_16', 'vgg13_fc_16_bn',
           'vgg13_fc_8', 'vgg13_fc_8_bn',
           'vgg13_fc_4', 'vgg13_fc_4_bn']

model_urls = {'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth'}


class VGG(nn.Module):

    def __init__(self, fc, features, output_channels=1000):
        super(VGG, self).__init__()

        self.expected_input_size = (224, 224)

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(8 * 7 * 7, fc),
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
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg11']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vgg13_fc_1024(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 1024
    model = VGG(make_layers(fully_connected_layer_size, cfg['B']), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vgg13_fc_1024_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 1024
    model = VGG(make_layers(fully_connected_layer_size, cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_fc_512(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 512
    model = VGG(make_layers(fully_connected_layer_size, cfg['B']), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vgg13_fc_512_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 512
    model = VGG(make_layers(fully_connected_layer_size, cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_fc_256(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 512
    model = VGG(make_layers(fully_connected_layer_size, cfg['B']), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vgg13_fc_256_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 512
    model = VGG(make_layers(fully_connected_layer_size, cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_fc_128(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 128
    model = VGG(make_layers(fully_connected_layer_size, cfg['B']), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vgg13_fc_128_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 128
    model = VGG(make_layers(fully_connected_layer_size, cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_fc_64(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 64
    model = VGG(make_layers(fully_connected_layer_size, cfg['B']), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vgg13_fc_64_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 64
    model = VGG(make_layers(fully_connected_layer_size, cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_fc_32(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 32
    model = VGG(make_layers(fully_connected_layer_size, cfg['B']), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vgg13_fc_32_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 32
    model = VGG(make_layers(fully_connected_layer_size, cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_fc_16(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 16
    model = VGG(make_layers(fully_connected_layer_size, cfg['B']), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vgg13_fc_16_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 16
    model = VGG(make_layers(fully_connected_layer_size, cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_fc_8(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 8
    model = VGG(make_layers(fully_connected_layer_size, cfg['B']), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vgg13_fc_8_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 8
    model = VGG(make_layers(fully_connected_layer_size, cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model

def vgg13_fc_4(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 4
    model = VGG(make_layers(fully_connected_layer_size, cfg['B']), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vgg13_fc_4_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    fully_connected_layer_size = 4
    model = VGG(make_layers(fully_connected_layer_size, cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model