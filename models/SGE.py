import torch.nn as nn
from torchvision.models import ResNet

import math, os
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch_dct as dct
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

__all__ = ['sge_resnet18', 'sge_resnet34', 'sge_resnet50', 'sge_resnet101',
           'sge_resnet152']
model_urls = {
    "sge_resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "sge_resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "sge_resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth"
}


class SpatialGroupEnhance(nn.Module):
    def __init__(self, planes, groups = 16):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()     # 64 x 64 x 56 x 56
        x = x.view(b * self.groups, -1, h, w)    # 1024 x 4 x 56 x 56
        xn = x * self.avg_pool(x)                # 1024 x 4 x 56 x 56   channel attention
        xn = xn.sum(dim=1, keepdim=True)         # 通道维度求和
        t = xn.view(b * self.groups, -1)         # 1024 x 56*56 resize到二维
        t = t - t.mean(dim=1, keepdim=True)      # Normalization
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)         # 64 x 16 x 56 x 56
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)

        # filecount=1
        # if h == w == 14:
        #     # for root, dir, files in os.walk('./heatmap/att/'):
        #     #     filecount = len(files)
        #     #     print(filecount)
        #
        #     for i in range(t.size(0)):
        #         sget = colorize(t[i][0])
        #         sget = sget.cpu().detach().numpy()
        #         plt.figure()
        #         plt.xticks([])
        #         plt.yticks([])
        #         plt.axis('off')
        #         plt.imshow(sget[0])
        #         plt.savefig('./heatmap/att/heatmap_sge{}_{}.jpg'.format(filecount//16, i))
        #         plt.show()
        #
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

    # def gauss(self, x, a, b, c):
    #     return torch.exp(-torch.pow(torch.add(x, -b), 2).div(2 * c * c)).mul(a)
    #
    # def colorize(self, x):
    #     ''' Converts a one-channel grayscale image to a color heatmap image '''
    #     if x.dim() == 2:
    #         cl = torch.unsqueeze(x, 0)
    #     if x.dim() == 3:
    #         cl = torch.zeros([3, x.size(1), x.size(2)])
    #         cl[0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    #         cl[1] = gauss(x, 1, .5, .3)
    #         cl[2] = gauss(x, 1, .2, .3)
    #         cl[cl.gt(1)] = 1
    #     elif x.dim() == 4:
    #         cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
    #         cl[:, 0, :, :] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    #         cl[:, 1, :, :] = gauss(x, 1, .5, .3)
    #         cl[:, 2, :, :] = gauss(x, 1, .2, .3)
    #     return cl

    # def dct_base(self, x):
class SpatialFreqGroupEnhance(nn.Module):
    def __init__(self, planes, groups = 16):
        super(SpatialFreqGroupEnhance, self).__init__()
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.radiu = 8
        self.A, self.mask = self.get_dct_filter(c2wh[planes], groups)
        # self.register_buffer('A','mask', self.get_dct_filter(c2wh[planes]))
        # self.dct = MultiSpectralDCTLayer(c2wh[planes], groups)

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()     # 64 x 64 x 56 x 56
        x = x.view(b * self.groups, -1, h, w)    # 1024 x 4 x 56 x 56
        xf = self.A * x * self.A.T
        xf = torch.mul(xf, self.mask)
        # xf = dct.dct_2d(x)
        # for i in range(self.radiu):
        #     mul = h // 7
        #     for xx in range(h):
        #         for yy in range(w):
        #             d = math.sqrt(xx^2 + yy^2)
        #             if (d >= i * mul) & (d < (i + 1) * mul):
        #                 continue
        #             else:
        #                 xf[:, i * mul : (i+1)*mul, xx, yy] = torch.tensor(1e-8)
        # xn = x * self.avg_pool(x)                # 1024 x 4 x 56 x 56   channel attention
        xn = xf.sum(dim=1, keepdim=True)         # 通道维度求和
        t = xn.view(b * self.groups, -1)         # 1024 x 4*56*56 resize到二维
        t = t - t.mean(dim=1, keepdim=True)      # Normalization
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)         # 64 x 16 x 56 x 56
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

    def get_dct_filter(self, size, groups):
        dct_filter = torch.zeros(size, size)
        A2D = torch.zeros(size, size)
        A = torch.zeros(groups, size, size)
        mul = size // groups
        for i in range(size):
            for j in range(size):
                if i == 0:
                    a = math.sqrt(1/size)
                else:
                    a = math.sqrt(2/size)
                dct_filter[i, j] = a * math.cos(math.pi * i / size * (j + 0.5))
                A2D[i, j] = math.sqrt(i**2 + j**2) // mul
        A[i, :, :] = [A2D.equal(torch.tensor(i)) for i in range(groups)]
        return dct_filter, A


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, size, groups):
        super(MultiSpectralDCTLayer, self).__init__()

        self.num_freq = groups

        self.register_buffer('weight', self.get_dct_filter(size))
        self.radiu = groups
        self.size = size

    def forward(self, x):

        # x = self.weight * x * self.weight.T
        x = x.cpu().detach().numpy()
        x = dct(x)

        for i in range(self.radiu):
            mul = self.size // 8
            for xx in range(self.size):
                for yy in range(self.size):
                    d = math.sqrt(xx^2 + yy^2)
                    if (d >= i * mul) & (d < (i + 1) * mul):
                        continue
                    else:
                        x[:, i, xx, yy] = torch.tensor(1e-8)

        # result = torch.sum(x, dim=[2, 3])       # 64 X 64 B X C
        return x

    def get_dct_filter(self, size):
        dct_filter = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                if i == 0:
                    a = math.sqrt(1/size)
                else:
                    a = math.sqrt(2/size)
                dct_filter[i, j] = a* math.cos(math.pi * i / size * (j + 0.5))
        return dct_filter


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.sge    = SpatialGroupEnhance(planes, 16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sge(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.sge    = SpatialGroupEnhance(64)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sge(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels=3, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x  = self.layer2(x)
        x  = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def _sgenet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print('Load pretrained params from url {}'.format(arch))
        # state_dict = load_state_dict_from_url(model_urls[arch], map_location='cpu', progress=True)
        state_dict = torch.load('./models/resnet18_msceleb.pth', map_location='cpu')['state_dict']
        curr_model_static_dict = model.state_dict()
        for name, param in state_dict.items():
            if "att" in name:
                continue
            else:
                curr_model_static_dict[name].copy_(param)
        model.load_state_dict(curr_model_static_dict)
    return model


def sge_resnet18(num_classes=7, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _sgenet('sge_resnet18', BasicBlock, [2, 2, 2, 2], pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)
    return  model

def sge_resnet34(num_classes=7, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def sge_resnet50(num_classes=7, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _sgenet('sge_resnet50', Bottleneck, [3, 4, 6, 3], pretrained=pretrained)
    model.fc = nn.Linear(2048, num_classes)
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)
    return model


def sge_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def sge_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def gauss(x, a, b, c):
    return torch.exp(-torch.pow(torch.add(x, -b), 2).div(2 * c * c)).mul(a)


def colorize(x):
    ''' Converts a one-channel grayscale image to a color heatmap image '''
    if x.dim() == 2:
        cl = torch.unsqueeze(x, 0)
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
        cl[1] = gauss(x, 1, .5, .3)
        cl[2] = gauss(x, 1, .2, .3)
        cl[cl.gt(1)] = 1
    elif x.dim() == 4:
        cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
        cl[:, 0, :, :] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
        cl[:, 1, :, :] = gauss(x, 1, .5, .3)
        cl[:, 2, :, :] = gauss(x, 1, .2, .3)
    return cl

if __name__=='__main__':
    model = sge_resnet18(pretrained=False)
    img = cv2.imread('../data/train_09115_aligned.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = torch.tensor(gray)
    gray = colorize(gray)
    gray = gray[0].detach().numpy()
    input = torch.randn(64, 3, 224, 224)
    out = model(input)
    print(out.size())
    input = colorize(input[0, 0])
    y = input.detach().numpy()[0]
    plt.figure()
    plt.imshow(gray)
    # sns_plot = sns.heatmap(gray, cmap='RdBu_r', xticklabels=8, yticklabels=8)
    plt.show()


