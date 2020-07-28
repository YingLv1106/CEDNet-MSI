import torch
from torch import nn

def conv1(in_chsnnels, out_channels):
    "1x1 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=1, stride=1, bias=False)

class ASPP(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(conv1(in_planes, out_planes),
                                   nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))

        self.conv3_4 = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                                               dilation=4, padding=4, bias=False),
                                     nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True)
                                     )

        self.conv3_8 = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                                                 dilation=8, padding=8, bias=False),
                                     nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True)
                                    )

        self.conv3_12 = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                                                 dilation=12, padding=12, bias=False),
                                     nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True)
                                    )


        self.avg = nn.AdaptiveAvgPool2d((28, 28))
        self.conv_avg = nn.Sequential(conv1(in_planes, out_planes),
                                   nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))
        self.conv_out= nn.Sequential(conv1(out_planes*5, out_planes),
                                   nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3_4(x)
        x3 = self.conv3_8(x)
        x4 = self.conv3_12(x)
        # x5 = self.image_pooling(x1)
        x5 = self.avg(x)#.expend_as(x)
        x5 = self.conv_avg(x5)
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out =self.conv_out(out)
        return out   #256*5
# a = torch.zeros((2,64,1))
# a= a.expand_as((2,64,224,224))
# print(a)