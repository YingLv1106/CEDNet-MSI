import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

def conv1(in_chsnnels, out_channels):
    "1x1 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=1, stride=1, bias=False)

def conv3(in_chsnnels, out_channels):
    "3x3 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)


class vgg16_bn(nn.Module):
    def __init__(self):
        super(vgg16_bn, self).__init__()
        self.vgg_pre = []
        #vggnet
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 1/2
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 1/4

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)  # 1/8

        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1, padding=1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1, padding=1)
        )


        self.features = nn.ModuleList(self.vgg_pre)


        self.__copy_param()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        # print(c3.size(),c4.size(),c5.size())
        return torch.cat((c3, c4, c5), dim=1)

    def __copy_param(self):

        # Get pretrained vgg network
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)

        # Concatenate layers of generator network
        DGG_features = list(self.conv1.children())
        DGG_features.extend(list(self.conv2.children()))
        DGG_features.extend(list(self.conv3.children()))
        DGG_features.extend(list(self.conv4.children()))
        DGG_features.extend(list(self.conv5.children()))
        DGG_features = nn.Sequential(*DGG_features)

        # Copy parameters from vgg16
        for layer_1, layer_2 in zip(vgg16.features, DGG_features):
            if(isinstance(layer_1, nn.Conv2d) and
               isinstance(layer_2, nn.Conv2d)):
                assert layer_1.weight.size() == layer_2.weight.size()
                assert layer_1.bias.size() == layer_2.bias.size()
                layer_2.weight.data = layer_1.weight.data
                layer_2.bias.data = layer_1.bias.data
        return




class deconv(nn.Module):
    def __init__(self):
        super(deconv, self).__init__()

        self.de1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.de2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.de3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.de4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.de5 = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.de1(x)
        out = self.de2(out)
        out = self.de3(out)
        out = self.de4(out)
        out = self.de5(out)
        return out

from models.ASPP import ASPP
class CEDANet(nn.Module):
    def __init__(self):
        super(CEDANet, self).__init__()
        self.rgb_vgg = vgg16_bn()

        self.aspp = ASPP(256+512+512, 256)

        self.deconv = deconv()
    def forward(self, rgb):
        out = self.rgb_vgg(rgb)
        out = self.aspp(out)
        # print(out.size())
        out = self.deconv(out)
        return out

