import torch
import torch.nn as nn

from torchvision.models.vgg import VGG
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.models.inception import Inception3
from typing import cast

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

###################################################################################
# ResNet34 https://arxiv.org/pdf/1512.03385.pdf 
###################################################################################

class ResNet34(ResNet):
    def __init__(self, dataset):
        self.name = 'resnet34'
        self.dataset = dataset 
        super().__init__(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=10)
        if dataset in ['mnist', 'fashion_mnist']:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  

###################################################################################
# VGG-11 https://arxiv.org/pdf/1409.1556.pdf
###################################################################################

class VGG11(VGG):
    def __init__(self, dataset):
            self.name = 'vgg11'
            self.dataset = dataset
            super().__init__(make_layers(vgg11_config, batch_norm=False), num_classes=10, dropout=0.5)
            if dataset in ['mnist', 'fashion_mnist']:
                self.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
                
vgg11_config = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

def make_layers(config, batch_norm=False):
    layers = []
    in_channels = 3
    for v in config:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

###################################################################################
# GoogLeNet https://arxiv.org/pdf/1409.4842.pdf
###################################################################################

class GoogLeNet(nn.Module):
    def __init__(self, dataset, num_classes=10, aux_logits=True):
        self.name = 'googlenet'
        self.dataset = dataset 
        super(GoogLeNet, self).__init__()
        
        self.aux_logits = aux_logits

        if dataset in ['mnist', 'fashion_mnist']:
            self.conv1 = ConvBlock(1, 64, kernel_size=7, stride=2, padding=3)
        else:
            self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1_pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 

###################################################################################
# Inception-v3 https://arxiv.org/pdf/1512.00567.pdf
###################################################################################

class InceptionV3(Inception3):
    def __init__(self, dataset):
        self.name = 'inception_v3'
        self.dataset = dataset 
        super().__init__(num_classes=10)
        if dataset in ['mnist', 'fashion_mnist']:
            self.Conv2d_1a_3x3 = ConvBlock(1, 32, kernel_size=3, stride=2)

